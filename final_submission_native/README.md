# Native Linux Git Secret Detector

This submission contains the native Git pre-commit detector and its frozen CodeBERT–Random Forest model. It does not include the Streamlit interface or Docker deployment. Research scripts and supporting results are kept in `research_evidence/`.

## Linux Mint installation

Requirements: Linux Mint, Git, Python 3.12 with `venv`, and network access to install the pinned Python wheels. The CodeBERT checkpoint and tokenizer are bundled; the installer does not download a model.

From this directory, install the runtime and global hook:

```sh
./native_app/install.sh
```

The installer creates `native_app/.venv`, installs CPU PyTorch and pinned inference requirements, verifies the frozen classifier and bundled CodeBERT snapshot, then sets the user-level `core.hooksPath` to the detector's hook directory. The application folder must stay at its installed path because the hook uses that path to load the model and Python environment.

The global installer refuses to replace an existing global or system `core.hooksPath`, or an existing global pre-commit hook. It does not modify repository-local Git configuration. A repository-local `core.hooksPath` overrides Git's global setting, so the global detector will not run in repositories that define one. Existing default `.git/hooks/pre-commit` hooks are chained before the detector; if one blocks, the detector does not override that decision.

To remove the global hook and its setting, run:

```sh
sh native_app/hook/uninstall_global.sh
```

## Try a commit

After installation, a regular commit in any repository that does not override
`core.hooksPath` runs the scanner automatically. For a harmless end-to-end
check, use a disposable repository:

```sh
mkdir -p /tmp/githook-demo
cd /tmp/githook-demo
git init
git config user.name "Githook Demo"
git config user.email "githook-demo@example.invalid"
printf 'answer = 42\n' > demo.py
git add demo.py
git commit -m "Try the Githook scanner"
```

The prediction is model-dependent, so even this harmless example can be
flagged. A message such as `COMMIT ALLOWED` means no positive prediction was
made for the eligible staged additions; `COMMIT BLOCKED` means at least one
addition was classified as a potential secret. The hook reports file paths and
line numbers, never candidate values. Do not use a real credential to test it.

## Runtime behavior

The hook scans eligible **staged added lines** from the Git index. It does not scan unstaged worktree edits, deletions, or unchanged surrounding context. Binary files are skipped; invalid encodings, unresolved index entries, size/count limits, model errors, malformed predictions, or service errors fail closed. A positive model prediction blocks the commit. A clean result means no positive predictions among eligible staged additions; it is not a guarantee that a repository contains no secrets.

The detector uses `microsoft/codebert-base` and tokenizer revision `3b0952feddeffad0063f274080e3c23d75e7eb39`, a 512-token limit, attention-mask-weighted mean pooling over the final hidden states, 768 float32 features, batch size 2, and CPU inference. The serialized Random Forest checksum and configuration are verified before inference. The service loads lazily over a private Unix socket, reuses the model while running, and shuts down after the configured 600-second idle interval.

## Reproducing model training

Training is separate from installing or using the Git hook. The frozen model is
already included for inference. The training CSVs and embedding caches are not
included in this submission, so training requires a separate research workspace
with these files, relative to that workspace:

```text
data/value_disjoint/train.csv
data/value_disjoint/validation.csv
data/value_disjoint/test.csv
data/test.csv
```

The script is `research_evidence/scripts/run_experiment.py`. It selects among
Random Forest candidates using the value-disjoint training and validation
splits, then in full mode evaluates the selected model on the value-disjoint and
original test sets. The test results have already been examined; do not use
them to choose settings or tune a later model. The training script does not
overwrite the packaged frozen classifier.

From the submission directory, set paths to this submission and to the
workspace containing those CSVs, then create a separate training environment:

```sh
SUBMISSION_DIR=/path/to/final_submission_native
RESEARCH_DATA_ROOT=/path/to/workspace-with-the-required-data
cd "$RESEARCH_DATA_ROOT"
python3.12 -m venv .venv-training
. .venv-training/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "$SUBMISSION_DIR/native_app/requirements-model.txt"
```

Run the small smoke experiment first. It uses a small train/validation sample
and writes to a fresh output directory:

```sh
python "$SUBMISSION_DIR/research_evidence/scripts/run_experiment.py" \
  --mode smoke --batch-size 2 --output-dir artifacts/githook_smoke
```

After reviewing the smoke run and confirming adequate RAM and disk space, a
full run can be started with a different, previously unused output directory:

```sh
python "$SUBMISSION_DIR/research_evidence/scripts/run_experiment.py" \
  --mode full --batch-size 2 --output-dir artifacts/githook_full_run
```

The encoder revision is pinned in the script. Transformers must be able to
access that checkpoint through its local cache or the network. Full mode is a
long CPU workload and includes test evaluation after model selection; it is
not required for ordinary hook operation.

## Troubleshooting

- If installation reports a conflicting `core.hooksPath`, review that existing
  setting manually; the installer intentionally refuses to replace it.
- If the hook does not run in one repository, check whether that repository
  defines its own `core.hooksPath`; a repository-level setting takes
  precedence over the global setting.
- If installation cannot create the virtual environment or install pinned
  dependencies, confirm Python 3.12, `venv`, disk space, and network access.
- If a commit is blocked with `SCAN INCOMPLETE`, the scan failed and the commit
  was stopped. Check the installation and local service permissions, then run
  `native_app/.venv/bin/python native_app/hook/pre_commit.py --verify-model`.
  Do not bypass the hook for a real credential; resolve the installation or
  model error first.

## Layout

- `native_app/`: standalone hook, local inference service, configuration, frozen Linux model, pinned encoder/tokenizer assets, and global installer/uninstaller.
- `research_evidence/`: training/split/audit scripts, original metric records, native functional evidence, runtime measurements, and Windows reproduction evidence. It is not imported by the runtime.

The Windows reproduction's large embedding and chunk archives are not included. Its recorded experiment, separate fitted forest, comparison report, and retained audit are included; those metrics cannot be recomputed from the included Windows files alone.

## Verification

The frozen model and local tokenizer can be checked without scanning a repository:

```sh
native_app/.venv/bin/python native_app/hook/pre_commit.py --verify-model
```

Run the scanner's controlled unit tests with:

```sh
PYTHONPATH=native_app/tests/hook native_app/.venv/bin/python -m unittest test_pre_commit.PreCommitTests test_global_install -v
```

For the complete packaged test suite, including the functional fixtures, run:

```sh
native_app/.venv/bin/python -m unittest discover -s native_app/tests/hook -p 'test_*.py' -v
```
