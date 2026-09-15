# Interview Guide

This is for you, not a recruiter — a way to actually understand this
project well enough to defend it in conversation, not a script to
memorize. Each answer explains the reasoning so you can handle a follow-up
that doesn't match the exact wording here.

---

### 1. What problem does this solve?

**SHORT ANSWER:** FraudLens takes a batch of credit-card transactions,
figures out which ones are worth a human's attention for possible fraud,
and does it in a way that's actually auditable — every prediction traces
back to a specific model version, a specific decision threshold, and a
specific pipeline run. It's less "detect fraud" and more "build the
plumbing that makes fraud detection trustworthy and operable."

**DEEPER EXPLANATION:** A model that outputs a probability is only useful
if there's a defensible process around it: data has to be validated before
it reaches the model, the model's threshold has to be chosen deliberately
(not just "0.5 because that's the default"), and someone reviewing a flagged
transaction needs to see the same numbers the model saw. That's the gap
between a notebook that prints an AUC and a system someone could actually
run a fraud-review queue against.

**POSSIBLE FOLLOW-UP:** "What would you need to add for this to be a real
production system?" — real-time/streaming scoring, drift monitoring,
auth on the dashboard, and validation against the real dataset instead of
the synthetic sample. All listed honestly in the README's Limitations.

---

### 2. Walk me through the architecture.

**SHORT ANSWER:** Raw CSV → ingestion (structural checks, a manifest) →
validation (row-level rules, valid/rejected split) → a single sklearn
pipeline that does feature engineering and scoring together → Postgres →
a Streamlit dashboard that only ever reads from Postgres. Airflow
orchestrates it with two DAGs: one for training (manual trigger), one for
daily inference.

**DEEPER EXPLANATION:** The one non-obvious design choice: every stage
reads from and writes to a known disk location instead of passing objects
in memory between functions. That's not an accident — Airflow's
`LocalExecutor` runs tasks that don't share memory, so if I'd written this
as "one big function that passes DataFrames around," it would've worked
locally and then needed a rewrite the moment I orchestrated it. Building
it disk-boundary-first meant the orchestration layer was almost free.

**POSSIBLE FOLLOW-UP:** "Why Postgres and not just files?" — see Q13.

---

### 3. Why is fraud detection difficult?

**SHORT ANSWER:** Two compounding problems: fraud is extremely rare (under
1% of transactions, even rarer in the real dataset at ~0.17%), and the
cost of a false negative (missed fraud) versus a false positive (annoying
a legitimate customer) aren't remotely equal. Both of those break the
naive approach of "train a classifier, optimize accuracy."

**DEEPER EXPLANATION:** With 99.5%+ of labels being "legitimate," a model
can get almost everything "right" by predicting the majority class and
learning nothing. And even a good model forces a real tradeoff: catch more
fraud by lowering the threshold, and you also block more real customers.
There's no threshold that makes both numbers good simultaneously — that's
why threshold selection in this project is explicit and configurable
instead of hardcoded.

**POSSIBLE FOLLOW-UP:** "How would you handle concept drift — fraud
patterns changing over time?" — I'd monitor the distribution of
`fraud_probability` over time and alert on a shift, which isn't
implemented here (see README > Future Improvements) but `model_runs`
already stores enough history to build that comparison.

---

### 4. Why is accuracy misleading here?

**SHORT ANSWER:** On this project's own synthetic sample, a classifier
that always predicts "legitimate" scores 99.50% accuracy while catching
zero fraud — that number is in
`notebooks/01_exploration.ipynb`, computed, not asserted. Accuracy rewards
getting the majority class right, and the majority class is the
uninteresting one.

**DEEPER EXPLANATION:** This is why the whole project leads with
precision, recall, F1, and PR-AUC instead. PR-AUC specifically summarizes
the precision/recall tradeoff across every threshold, which is what you
actually care about when the decision threshold itself is a tunable
choice — see Q6.

**POSSIBLE FOLLOW-UP:** "What if accuracy were 99.9% and recall were only
50%?" — that means half of all fraud goes undetected while the topline
number looks great; exactly the trap this project is built to avoid
reporting.

---

### 5. Precision vs. recall.

**SHORT ANSWER:** Precision: of the transactions the model flagged, what
fraction were actually fraud? Recall: of the transactions that were
actually fraud, what fraction did the model catch? On the champion model's
test set here: precision 0.808, recall 0.875 — it catches 87.5% of fraud,
and when it flags something, it's right about 81% of the time.

**DEEPER EXPLANATION:** They trade off against each other via the
threshold: flag more things (lower the threshold) and recall goes up while
precision goes down, because you catch more real fraud but also more false
alarms. `scripts/optimize_threshold.py` makes that tradeoff visible as a
table instead of hiding it behind a single default number.

**POSSIBLE FOLLOW-UP:** "Which matters more for fraud?" — depends on the
business's cost tolerance, which is exactly what the illustrative cost
model in `config/settings.yaml` (see the model card) is meant to make
explicit rather than assumed.

---

### 6. What is PR-AUC?

**SHORT ANSWER:** The area under the precision-recall curve — one number
summarizing how good the precision/recall tradeoff is across every
possible threshold, without committing to any single one. It's the metric
this project uses to pick the champion model, instead of accuracy or even
a single F1 score.

**DEEPER EXPLANATION:** ROC-AUC (the more commonly-known cousin) can look
deceptively good on highly imbalanced data, because the false-positive
*rate* it's based on divides by the huge number of true negatives, making
even a mediocre model's ROC curve look strong. PR-AUC doesn't have that
denominator problem — it's much more sensitive to how the model performs
on the rare positive class, which is exactly the class that matters here.
Concretely in this project: the dummy baseline's PR-AUC (0.0049) sits
almost exactly at the dataset's actual fraud rate — that's the
mathematical floor for a model with zero signal, and a good sanity check
that PR-AUC is measuring something real.

**POSSIBLE FOLLOW-UP:** "Why not just use F1?" — F1 is PR tradeoff at *one*
threshold; PR-AUC evaluates the model before you've committed to a
threshold, which is the right stage to be comparing models at.

---

### 7. Why Logistic Regression?

**SHORT ANSWER:** As the simplest real baseline above the dummy classifier
— fast to train, easy to explain via coefficients, and a useful
sanity check for whether the problem is even close to linearly separable
before reaching for more complex models.

**DEEPER EXPLANATION:** It ended up being the champion here (with SMOTE),
which is genuinely a property of how the *synthetic* sample was
constructed — fraud rows get a mean/variance shift on a handful of
features, which is close to linearly separable by design. I say that
explicitly in the model card rather than letting the result imply
something false about real fraud data, where nonlinear interactions
usually make tree ensembles win.

**POSSIBLE FOLLOW-UP:** "So is Logistic Regression the right choice for
production?" — Only provisionally, pending the same comparison against
the real dataset; the whole point of running the comparison in code
instead of assuming an answer is that the answer can change.

---

### 8. Why Random Forest?

**SHORT ANSWER:** A bagged tree ensemble that captures nonlinear feature
interactions Logistic Regression can't, and is fairly robust to
hyperparameter choices out of the box — a reasonable "does a nonlinear
model help here" checkpoint.

**DEEPER EXPLANATION:** On this dataset it actually came in with the
*lowest* PR-AUC of the non-dummy models (0.929) despite perfect precision
at the default threshold — it was more conservative, missing more real
fraud (recall 0.526 at threshold 0.5) in exchange for never crying wolf. A
worthwhile result specifically because it wasn't the best: it shows the
comparison is real, not decided in advance.

**POSSIBLE FOLLOW-UP:** "Why did Random Forest underperform here?" — most
likely because the synthetic data's signal is close to linear, which plays
to Logistic Regression's strength and doesn't give tree splits much
nonlinear structure to exploit.

---

### 9. Why XGBoost?

**SHORT ANSWER:** Gradient boosting is the standard strong baseline for
tabular classification in general, and for fraud detection specifically —
worth including even though (or especially because) it didn't win here, so
the project doesn't just say "we used XGBoost" as an unexamined default.

**DEEPER EXPLANATION:** `scale_pos_weight` handles the class imbalance
natively (set to the train-fold's negative/positive ratio), without
needing SMOTE. It landed second-best by PR-AUC (0.930) behind the SMOTE
Logistic Regression champion (0.961) — close, and on the real dataset with
its more genuinely nonlinear fraud patterns, I'd expect this ranking to be
a much closer call, if not reversed.

**POSSIBLE FOLLOW-UP:** "Would you tune hyperparameters further?" — yes,
with more time; current settings (`n_estimators=300, max_depth=5`) are
reasonable defaults, not the result of a tuning search — I say so rather
than implying they were optimized.

---

### 10. How did you handle class imbalance?

**SHORT ANSWER:** Two strategies, measured against each other rather than
assumed: class weighting (the default for every model — no extra
dependency, can't leak) and SMOTE oversampling, compared head-to-head on
Logistic Regression specifically. SMOTE won by a small but real margin
(validation PR-AUC 0.9605 vs. 0.9565).

**DEEPER EXPLANATION:** I didn't want to just claim "we used SMOTE because
that's what fraud tutorials do" — the comparison table exists specifically
so the claim is backed by a number I actually generated. Undersampling was
considered but not implemented; with only ~500 fraud examples even in the
full real dataset, throwing away majority-class rows felt like a worse
tradeoff than either weighting or oversampling.

**POSSIBLE FOLLOW-UP:** "What if SMOTE had made things worse?" — then the
comparison table would show that, and I'd have shipped class weighting
instead. That's the point of measuring instead of assuming.

---

### 11. Why must SMOTE only touch training data?

**SHORT ANSWER:** SMOTE generates synthetic minority-class examples by
interpolating between real ones. If it runs before the train/test split
(or on the validation/test folds), the model can effectively be evaluated
on synthetic points that are close neighbors of points it trained on —
inflating the reported score without the model actually generalizing
better.

**DEEPER EXPLANATION:** This is exactly why SMOTE lives inside an
`imblearn.Pipeline` (not scikit-learn's own `Pipeline`, which doesn't
support a resampling step that changes the number of rows) that's fit only
on the training fold. Cross-validation would need the same care — SMOTE
inside each fold, not before folding.

**POSSIBLE FOLLOW-UP:** "Is this the same category of mistake as scaling
before splitting?" — yes, same root cause (fitting something on data the
model will later be evaluated against) with a more dangerous consequence,
since SMOTE literally manufactures new rows rather than just re-centering
existing ones.

---

### 12. What is data leakage?

**SHORT ANSWER:** Any way information from outside the training set
(especially from validation/test data, or from the future) influences
model training, making evaluation metrics look better than the model will
actually perform on genuinely new data.

**DEEPER EXPLANATION:** Three places this project deliberately guards
against it: (1) the feature `ColumnTransformer` is fit only on the
training fold, then reused unchanged — never refit on validation or test
data; (2) SMOTE runs training-fold-only, per Q11; (3) the test set is
scored exactly once, after the champion and threshold are already chosen
on validation — never used to pick between models, or the leakage is just
moved from "the model saw the data" to "the human picking the model saw
the test score and chose accordingly."

**POSSIBLE FOLLOW-UP:** "What about the stratified split — could that
leak information?" — no; stratification only balances the *label ratio*
across splits, it doesn't let any row's features influence another split.

---

### 13. Why PostgreSQL?

**SHORT ANSWER:** The dashboard needs to filter, sort, join, and aggregate
transaction and prediction data interactively — "show me the 20
highest-risk transactions from the last hour" is a `WHERE` + `ORDER BY` +
`LIMIT` in SQL and a full-file re-scan in a CSV. A relational store fits
the actual access pattern.

**DEEPER EXPLANATION:** The schema deliberately excludes `V1`-`V28` — the
app layer (dashboard, predictions) never consumes the raw anonymized
features directly, only the probability they produced, so storing them in
Postgres would be paying write and storage cost for columns nothing reads.
They stay in `data/processed/` as the model's feature-store input instead.

**POSSIBLE FOLLOW-UP:** "Why not just query the CSVs directly from
Streamlit?" — works for one page at demo scale, breaks down once you want
joins across transactions/predictions/model_runs, want the dashboard
decoupled from wherever the pipeline last ran, or want more than one
reader at a time.

---

### 14. Why S3?

**SHORT ANSWER:** Local disk is fine for a laptop, but it doesn't survive
past the lifetime of one machine or one EC2 instance. S3 is durable,
versionable storage for the raw transaction files that everything else
derives from — `STORAGE_MODE=s3` swaps the backend without any pipeline
code changing.

**DEEPER EXPLANATION:** `src/storage/s3.py` wraps boto3 behind
`upload_file`/`download_file`/`object_exists`, tested against a
moto-mocked bucket rather than real AWS, and never reads credentials from
config — boto3's standard credential chain (env vars, `~/.aws/credentials`,
or an IAM role) handles that, so nothing sensitive ever needs to live in
this repo's config files.

**POSSIBLE FOLLOW-UP:** "Why not store predictions in S3 too?" — different
access pattern: predictions get queried/filtered/joined, which is
Postgres's job; raw files get written once and read whole, which is S3's.

---

### 15. Why Airflow?

**SHORT ANSWER:** Five sequential steps that need to run on a schedule,
retry on transient failure, and have one place to see whether last night's
run succeeded — a cron job doing that with five separate scripts has none
of that visibility, and the failure of step 3 wouldn't stop steps 4 and 5
from running on bad input.

**DEEPER EXPLANATION:** Two DAGs, not one: training (manual trigger) and
daily inference are separated deliberately, because retraining shouldn't
happen automatically just because inference ran — a new model should be a
reviewed decision, not a side effect of a schedule. Also worth knowing:
Airflow itself only runs inside its own Docker image — it pulls a large,
strictly-pinned dependency tree that regularly conflicts with app-level
pandas/numpy/scikit-learn versions, so it's deliberately never installed
into the main venv or `requirements.txt`.

**POSSIBLE FOLLOW-UP:** "Why not just two cron jobs?" — you'd lose
retries, a visible run history, and the dependency graph between
ingest → validate → score that Airflow gives you for free; at some point
you're just reimplementing a worse version of an orchestrator.

---

### 16. Why Docker?

**SHORT ANSWER:** "Works on my machine" isn't good enough for a system
with five moving parts (Postgres, three Airflow processes, a dashboard) —
Docker Compose makes `docker compose up` reproduce the whole stack
identically on any machine with Docker installed.

**DEEPER EXPLANATION:** Two separate images on purpose: the root
`Dockerfile` builds a lean dashboard image (it only needs
`src/database`/`src/utils` and the DB driver, since the dashboard never
touches the model or raw data directly); `airflow/Dockerfile` extends the
official Airflow image with a thin extra-requirements layer. Keeping them
separate means a change to one doesn't force a rebuild of the other, and
the dashboard image stays small.

**POSSIBLE FOLLOW-UP:** "Why not one Dockerfile for everything?" — would
work, but couples the dashboard's dependencies to Airflow's, defeating the
whole reason Airflow gets its own image in the first place.

---

### 17. Why not Kafka initially?

**SHORT ANSWER:** Nothing in this problem needs sub-second latency — fraud
review is inherently a batch/near-real-time workflow, and a daily
(or hourly) Airflow-scheduled batch job is simpler to build, run, and
debug than a streaming pipeline, for the same outcome at this stage.

**DEEPER EXPLANATION:** Kafka earns its complexity when you actually need
to react to individual events within milliseconds-to-seconds, or need
multiple independent consumers of the same event stream. Neither is true
here yet. The codebase is structured so a streaming ingestion path could
be added later without touching validation, features, or the model — that
boundary already exists because ingestion is its own module with a clean
interface — but building it now would be solving a problem this project
doesn't have.

**POSSIBLE FOLLOW-UP:** "How would you add it later?" — a Kafka consumer
would replace `src/ingestion/ingest.py`'s file-reading step with
message-reading, hand rows to the same validation/feature/scoring code
unchanged, and write to the same `fraud_predictions` table — the
downstream pipeline wouldn't need to know ingestion changed.

---

### 18. How does threshold selection work?

**SHORT ANSWER:** `scripts/optimize_threshold.py` sweeps thresholds
0.05–0.95 on the *validation* split (never test) and reports
precision/recall/F1/false-positive-and-negative-counts/an illustrative
expected-cost figure at each one. A human reviews that output and updates
`config/settings.yaml`'s `decision_threshold` — the script doesn't do it
automatically.

**DEEPER EXPLANATION:** That's a deliberate human-in-the-loop gate, the
same way a promotion step would work in a real ML pipeline — a script
silently changing the production operating point is exactly the kind of
thing that should require a human to look at the tradeoff first. Also
worth knowing: the decision threshold (used for the binary `is_flagged`
column) is a completely separate concept from the dashboard's risk bands
(LOW/REVIEW/HIGH) — a transaction can show as HIGH risk without crossing
the decision threshold.

**POSSIBLE FOLLOW-UP:** "What did you actually pick, and why?" — 0.60,
because it's where both F1 and the illustrative expected-cost metric
happened to bottom out/peak together for the current champion — see the
model card for the full sweep table.

---

### 19. What happens when bad data arrives?

**SHORT ANSWER:** It gets rejected at validation, not silently dropped or
allowed to crash the pipeline — `src/validation/validator.py` checks
missing required values, invalid `Class`/`Amount`/`Time`, non-numeric
feature values, and duplicates, and splits input into
`transactions_valid.csv` and `transactions_rejected.csv` (with a specific
`rejection_reason` per row) instead of failing the whole batch over a few
bad rows.

**DEEPER EXPLANATION:** There's a second failure mode above row-level
badness: if the file is missing required *columns* entirely, that's not a
row problem, it's a structural one — `ValidationError` is raised and the
whole batch fails, correctly, rather than silently producing a
mostly-empty valid set. Both ingestion and validation log a
`pipeline_failed` event and write a `failed` status to `pipeline_runs`, so
the Pipeline Health dashboard page shows it.

**POSSIBLE FOLLOW-UP:** "How would you test this?" —
`tests/test_validation.py` covers exactly this: negative amounts, invalid
`Class` values, missing values, non-numeric features, duplicates, and
missing columns, each as a targeted case rather than one big fixture.

---

### 20. How would the architecture scale?

**SHORT ANSWER:** The batch pipeline scales by data volume (bigger
instance, or eventually partitioning the ingestion step), and by
frequency (Airflow already supports running more often than daily). The
part that would need real rework is going from "batch, once a day" to
"score every transaction as it happens" — that's a different architecture,
not just a bigger version of this one.

**DEEPER EXPLANATION:** Concretely, near-term scaling needs almost no
redesign: bigger Postgres instance (RDS), more Airflow worker capacity, a
bigger EC2 instance for the dashboard. What *would* need redesign for
true real-time scoring: a streaming ingestion path (see Q17), a model-serving
layer that isn't "load a joblib file and call predict_proba in a batch
task," and probably splitting the dashboard's read load onto a replica so
it doesn't compete with write-heavy ingestion.

**POSSIBLE FOLLOW-UP:** "What's the actual bottleneck at scale?" — almost
certainly Postgres write throughput on `fraud_predictions` once you're
inserting many thousands of rows per batch; batched inserts (already how
`repository.save_predictions` works, one `session.add_all` per call
rather than one insert per row) buy real headroom before that's a problem.

---

### 21. How would real-time streaming be added?

**SHORT ANSWER:** Swap the ingestion source from "read a CSV" to "consume
a Kafka topic," keep validation/features/scoring/persistence unchanged,
and move scoring from an Airflow batch task to a small always-on
consumer process. The model pipeline itself (`predict_proba` on a fitted
sklearn `Pipeline`) doesn't care whether its input arrived from a file or
a message.

**DEEPER EXPLANATION:** The reason this is a clean extension rather than a
rewrite: ingestion, validation, features, and prediction are already
separate modules with narrow interfaces (a DataFrame in, a DataFrame or
summary out), not one monolithic script. That separation is exactly what
this question is testing for.

**POSSIBLE FOLLOW-UP:** "What would break?" — the current design writes
one row to `fraud_predictions` per Airflow task run, which assumes
batch-sized writes; a streaming version would need to think about
per-message commit granularity and backpressure, neither of which the
current code handles.

---

### 22. What would you improve with more time?

**SHORT ANSWER:** In order: validate everything against the real dataset
instead of just the synthetic sample, add drift monitoring on the
prediction-probability distribution, and add a CI workflow that runs
`pytest` on every push — none of which are implemented here, and all of
which are called out explicitly in the README instead of implied to
exist.

**DEEPER EXPLANATION:** I'd also want a real feature store if features
ever got reused across more than one model, and a scheduled (not just
manually-triggered) retraining path gated on an actual drift signal rather
than a fixed calendar interval — retraining on a schedule regardless of
whether the model has actually degraded is a common but not obviously
correct default.

**POSSIBLE FOLLOW-UP:** "Why didn't you build [X] already?" — time and
scope tradeoffs for a portfolio project; the goal was depth on the core
pipeline (leakage-safe splitting, measured imbalance handling, a real
threshold analysis, a dashboard backed by a real schema) over breadth
across every possible production concern.
