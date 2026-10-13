# Publish the project to GitHub

The archive contains the source and Git history. Extract it to a new folder named
`fraudlens`; do not overwrite your only copy of the original project.

## 1. Run it locally

In WSL2, put it under your Linux home directory when possible, for example
`~/projects/fraudlens`. Open a terminal there and run the README quick start.
Run `pytest -q` and `ruff check .` before publishing.

## 2. Create an empty GitHub repository

Sign in to GitHub and create `fraudlens`. Start private while checking the build and
model claims; make it public when ready for your résumé. Do not initialize it with
a README, license, or .gitignore because those files/history already exist locally.
No license is assumed on your behalf.

## 3. Connect and push

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git status
git log --oneline -5
git remote -v
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraudlens.git
git push -u origin main
```

Authenticate through GitHub's browser/credential-manager flow when prompted.
Do not put a token in the repository URL or commit a token. If `origin` already
exists, inspect it first and use `git remote set-url origin ...` only if you intend
to replace that destination. Do not force-push to an existing repository.

## 4. Check Actions

Open the Actions tab. The workflow tests the code with PostgreSQL 16, builds both
images, and checks DAG imports. A workflow file is not a successful CI run: wait
for it to pass, and inspect any failure before describing the stack as verified.
Then run the manual Docker acceptance steps in `docs/runbook.md`.

## 5. Finish the résumé presentation

Add screenshots from your own successful dashboard run, a short demonstration
recording, and measured real-data results once available. Explain at least one
failure you reproduced and fixed. Do not present synthetic results as real-world
fraud performance, or a local deployment as an AWS deployment.

Before making your own commits, configure your name and the verified or privacy-protected
email address you use on GitHub:

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

For subsequent edits:

```bash
git add <the-files-you-changed>
git diff --cached
git commit -m "Describe the behavior changed"
git push
```

The supplied history retains original authorship and co-author attribution. There
is no need to rewrite it to present a technically credible project.
