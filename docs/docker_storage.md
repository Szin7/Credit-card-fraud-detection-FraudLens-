# Docker and Windows storage notes

The failed Airflow build ended during image export with `SIGBUS`; Docker subsequently reported I/O errors, an HTTP 500 from its Windows client, and missing WSL integration. These are infrastructure symptoms, not model training results.

## Before rebuilding

Check C: space in PowerShell:

```powershell
Get-PSDrive C
wsl --list --verbose
docker version
```

Don't attempt a large Airflow build until `docker version` returns both Client and Server details. On WSL, `docker compose ps` must work as well.

With around 16 GB free on C:, avoid another full local Airflow build. The optional GitHub Actions workflow can perform the container exercise on a hosted runner instead. Docker's image cache still needs routine care.

## Prefer another drive for Docker data

If an internal D: drive has enough free capacity and Docker Desktop launches normally, use **Docker Desktop → Settings → Resources → Advanced → Disk image location** to move its Linux disk image. Follow the application prompts and retain the existing image, not a new blank one. Do not move or delete `.vhdx` files manually. Docker documents this setting at https://docs.docker.com/desktop/settings-and-maintenance/settings/.

Move completed only after you confirm existing containers and database volumes are still present. Do not select an external removable drive as permanent Docker storage unless you accept the availability and performance risks.

## Avoid destructive cleanup

Safe inspection commands, once Docker works:

```bash
docker system df -v
docker buildx du
docker compose ps -a
```

Unused build caches may be selectively removed with `docker builder prune`; this can force future rebuilds. Do **not** use `docker compose down -v`, `docker volume prune`, `docker system prune -a --volumes`, or Docker Desktop factory reset while your other database and Kubernetes lab volumes are needed.

## Input data inside containers

The Compose setup mounts a volume at `/opt/fraudlens/data`. That mount hides the example CSV copied there by the Airflow Dockerfile. For a reliable training run, use the read-only `/incoming` bind mount from `data/incoming` and set `TRAINING_INPUT` and `INFERENCE_INPUT` to those paths.

Generate a small, clearly synthetic test input with:

```bash
python -m scripts.prepare_docker_demo
```

Set these in `.env` before starting Airflow:

```dotenv
TRAINING_INPUT=/incoming/docker-demo-train.csv
INFERENCE_INPUT=/incoming/docker-demo-inference.csv
DATASET_KIND=synthetic
TRAINING_SPLIT=stratified
```

These files are only a container acceptance fixture. The Kaggle real-data model remains separate from this test and isn't silently installed into Docker.
