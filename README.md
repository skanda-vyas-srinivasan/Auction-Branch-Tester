# Branch Learning Auction Distance Anchor

This repo is the clean auction-anchor rerun for the MILP distance transfer experiment.

The goal is to train a Learn2Branch-style GCNN branching policy on the official
combinatorial-auction benchmark family, then use that model as the anchor for a
cleaner distance-transfer study.

Why auctions:

- It is one of the official Learn2Branch benchmark families.
- The Maudet distance profile is cleaner from auctions than from set cover:
  - auctions to auctions: about 0
  - auctions to independent set: about 1
  - auctions to set cover: about 2
  - auctions to facility location: about 3

## Vast.ai Run

Run from the repo root on a Python 3.11 CUDA instance:

```bash
GITHUB_TOKEN=... python runners/vast_cauctions_runner.py
```

Useful staged runs:

```bash
# Generate instances/samples, archive samples, then stop.
L2B_STAGE=samples SAMPLE_ARCHIVE_DIR=/path/to/storage GITHUB_TOKEN=... python runners/vast_cauctions_runner.py

# Train after restoring a sample archive.
L2B_STAGE=train SAMPLE_ARCHIVE_PATH=/path/to/samples_cauctions_100_500.tar.gz GITHUB_TOKEN=... python runners/vast_cauctions_runner.py

# Run a small solve evaluation after training.
L2B_STAGE=evaluate GITHUB_TOKEN=... python runners/vast_cauctions_runner.py
```

The runner saves the trained model to:

```text
models/baseline_cauctions/train_params.pkl
models/baseline_cauctions/train_log.txt
```

Large generated samples and instances stay under `vast_work/` and are ignored by
git.
