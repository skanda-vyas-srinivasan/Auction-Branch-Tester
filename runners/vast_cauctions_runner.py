#!/usr/bin/env python3
"""Run the Learn2Branch/Ecole combinatorial-auction pipeline on Vast.ai.

This runner is intentionally auction-only. It trains a GCNN branching policy on
the official Learn2Branch combinatorial-auction distribution:

    100 items, 500 bids

Run from the repo root:

    GITHUB_TOKEN=mytoken python runners/vast_cauctions_runner.py

Stages:

    L2B_STAGE=samples    generate instances, collect samples, archive samples
    L2B_STAGE=train      restore samples if needed, train model
    L2B_STAGE=evaluate   run a small solve evaluation with trained weights
    L2B_STAGE=all        samples -> train -> evaluate
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path


ROOT = Path(os.environ.get("L2B_ROOT", Path(__file__).resolve().parents[1])).resolve()
WORK_DIR = ROOT / "vast_work"
ECOLE_REPO = WORK_DIR / "learn2branch-ecole"
ORIGINAL_REPO = WORK_DIR / "learn2branch"
ARTIFACT_DIR = WORK_DIR / "artifacts"

PROBLEM = "cauctions"
PROBLEM_FOLDER = "cauctions/100_500"
CHECKED_IN_MODEL_DIR = ROOT / "models" / "baseline_cauctions"
CHECKED_IN_RESULTS_DIR = ROOT / "results"

SEED = int(os.environ.get("L2B_SEED", "0"))
N_ITEMS = 100
N_BIDS = 500

N_TRAIN_INSTANCES = int(os.environ.get("L2B_N_TRAIN_INSTANCES", "10000"))
N_VALID_INSTANCES = int(os.environ.get("L2B_N_VALID_INSTANCES", "2000"))
N_TEST_INSTANCES = int(os.environ.get("L2B_N_TEST_INSTANCES", "2000"))
N_TRANSFER_INSTANCES = int(os.environ.get("L2B_N_TRANSFER_INSTANCES", "100"))

TRAIN_SAMPLES = int(os.environ.get("L2B_TRAIN_SAMPLES", "100000"))
VALID_SAMPLES = int(os.environ.get("L2B_VALID_SAMPLES", "20000"))
TEST_SAMPLES = int(os.environ.get("L2B_TEST_SAMPLES", "0"))
NODE_RECORD_PROB = float(os.environ.get("L2B_NODE_RECORD_PROB", "0.05"))
SAMPLE_TIME_LIMIT = float(os.environ.get("L2B_SAMPLE_TIME_LIMIT", "3600"))
N_JOBS = int(os.environ.get("L2B_N_JOBS", "32"))

MAX_EPOCHS = int(os.environ.get("L2B_MAX_EPOCHS", "1000"))
EPOCH_SAMPLES = int(os.environ.get("L2B_EPOCH_SAMPLES", "10000"))
BATCH_SIZE = int(os.environ.get("L2B_BATCH_SIZE", "16"))
PRETRAIN_BATCH_SIZE = int(os.environ.get("L2B_PRETRAIN_BATCH_SIZE", "32"))
VALID_BATCH_SIZE = int(os.environ.get("L2B_VALID_BATCH_SIZE", "32"))

SAMPLE_LOG_EVERY = int(os.environ.get("L2B_SAMPLE_LOG_EVERY", "100"))
SAMPLE_CHECKPOINT_EVERY = int(os.environ.get("L2B_SAMPLE_CHECKPOINT_EVERY", "5000"))

EVAL_INSTANCES_PER_SIZE = int(os.environ.get("L2B_EVAL_INSTANCES_PER_SIZE", "5"))
EVAL_SEEDS = int(os.environ.get("L2B_EVAL_SEEDS", "1"))
EVAL_TIME_LIMIT = float(os.environ.get("L2B_EVAL_TIME_LIMIT", "600"))
EVAL_NODE_LIMIT = int(os.environ.get("L2B_EVAL_NODE_LIMIT", "10000"))

SAMPLE_DIR = ECOLE_REPO / "data" / "samples" / PROBLEM_FOLDER
INSTANCE_DIR = ECOLE_REPO / "data" / "instances" / PROBLEM
MODEL_DIR = ECOLE_REPO / "model" / PROBLEM / str(SEED)
SAMPLE_ARCHIVE = ARTIFACT_DIR / "samples_cauctions_100_500.tar.gz"
SAMPLE_MANIFEST = ARTIFACT_DIR / "samples_cauctions_100_500_manifest.json"

GITHUB_REPOSITORY = os.environ.get(
    "GITHUB_REPOSITORY",
    "skanda-vyas-srinivasan/Auction-Branch-Tester",
)
REMOTE_URL = "https://{token}@github.com/{repo}.git"


def run(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(arg) for arg in args), flush=True)
    return subprocess.run(args, cwd=cwd, check=check)


def run_capture(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True)


def require_python_311() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "This runner requires Python 3.11.x for Ecole compatibility. "
            f"Current Python is {sys.version} at {sys.executable}."
        )


def pip_install_missing() -> None:
    required_modules = {
        "ecole": "ecole",
        "pyscipopt": "pyscipopt",
        "torch": "torch",
        "torch_geometric": "torch-geometric",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
    }
    missing = []
    for module_name, package_name in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        run([sys.executable, "-m", "pip", "install", "-U", *missing])
    else:
        print("python dependencies already installed", flush=True)


def print_system_info() -> None:
    print(f"root: {ROOT}", flush=True)
    print(f"python: {sys.executable} {sys.version}", flush=True)
    total, used, free = shutil.disk_usage(ROOT)
    print(
        f"disk: total={total / 1024**3:.1f}GB "
        f"used={used / 1024**3:.1f}GB free={free / 1024**3:.1f}GB",
        flush=True,
    )
    try:
        import torch

        print(f"torch: {torch.__version__}", flush=True)
        print(f"cuda available: {torch.cuda.is_available()}", flush=True)
        print(f"cuda version: {torch.version.cuda}", flush=True)
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as exc:
        print(f"torch inspection failed: {exc}", flush=True)

    run(["nvidia-smi"], check=False)


def git_setup() -> bool:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[git] GITHUB_TOKEN not set; skipping git push", flush=True)
        return False
    if not (ROOT / ".git").exists():
        print(f"[git] {ROOT} is not a git repo; skipping git push", flush=True)
        return False

    remote = REMOTE_URL.format(token=token, repo=GITHUB_REPOSITORY)
    remotes = run_capture(["git", "remote"], cwd=ROOT, check=False).stdout.splitlines()
    if "origin" in remotes:
        run(["git", "remote", "set-url", "origin", remote], cwd=ROOT)
    else:
        run(["git", "remote", "add", "origin", remote], cwd=ROOT)
    run(["git", "config", "user.email", os.environ.get("GIT_AUTHOR_EMAIL", "vast-runner@users.noreply.github.com")], cwd=ROOT)
    run(["git", "config", "user.name", os.environ.get("GIT_AUTHOR_NAME", "vast-runner")], cwd=ROOT)
    return True


def git_commit_push(message: str, paths: list[Path]) -> None:
    if not git_setup():
        return

    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        print(f"[git] no existing paths for {message}", flush=True)
        return

    rel_paths = [str(path.relative_to(ROOT)) for path in existing_paths]
    run(["git", "add", "-f", "--", *rel_paths], cwd=ROOT)
    status = run_capture(["git", "status", "--porcelain", "--", *rel_paths], cwd=ROOT)
    if not status.stdout.strip():
        print(f"[git] no changes to commit for {message}", flush=True)
        return

    run(["git", "commit", "-m", message], cwd=ROOT)
    run(["git", "push", "origin", "HEAD"], cwd=ROOT)
    print(f"[git] pushed {message}", flush=True)


def remove_nested_git_metadata(path: Path) -> None:
    nested = path / ".git"
    if nested.exists():
        shutil.rmtree(nested)


def ensure_repos() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not ECOLE_REPO.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ds4dm/learn2branch-ecole.git", str(ECOLE_REPO)])
        remove_nested_git_metadata(ECOLE_REPO)
    if not ORIGINAL_REPO.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ds4dm/learn2branch.git", str(ORIGINAL_REPO)])
        remove_nested_git_metadata(ORIGINAL_REPO)


def patch_scripts() -> None:
    dataset_py = ECOLE_REPO / "02_generate_dataset.py"
    train_py = ECOLE_REPO / "03_train_gnn.py"
    utilities_py = ECOLE_REPO / "utilities.py"

    text = dataset_py.read_text()
    if "--train-size" not in text:
        text = text.replace(
            "    parser.add_argument(\n"
            "        '-j', '--njobs',\n"
            "        help='Number of parallel jobs.',\n"
            "        type=int,\n"
            "        default=1,\n"
            "    )",
            "    parser.add_argument(\n"
            "        '-j', '--njobs',\n"
            "        help='Number of parallel jobs.',\n"
            "        type=int,\n"
            "        default=1,\n"
            "    )\n"
            "    parser.add_argument('--train-size', type=int, default=100000)\n"
            "    parser.add_argument('--valid-size', type=int, default=20000)\n"
            "    parser.add_argument('--test-size', type=int, default=20000)\n"
            "    parser.add_argument('--node-record-prob', type=float, default=0.05)\n"
            "    parser.add_argument('--time-limit', type=float, default=None)",
        )
        text = text.replace(
            "    train_size = 100000\n"
            "    valid_size = 20000\n"
            "    test_size = 20000\n"
            "    node_record_prob = 0.05",
            "    train_size = args.train_size\n"
            "    valid_size = args.valid_size\n"
            "    test_size = args.test_size\n"
            "    node_record_prob = args.node_record_prob",
        )
        text = text.replace(
            '    print(f"{len(instances_train)} train instances for {train_size} samples")',
            "    if args.time_limit is not None:\n"
            "        time_limit = args.time_limit\n\n"
            '    print(f"{len(instances_train)} train instances for {train_size} samples")',
        )
        text = text.replace(
            "collect_samples(instances_valid, out_dir + '/valid', rng, test_size,",
            "collect_samples(instances_valid, out_dir + '/valid', rng, valid_size,",
        )

    if "existing_samples = len(glob.glob(f'{out_dir}/sample_*.pkl'))" not in text:
        text = text.replace(
            "    os.makedirs(out_dir, exist_ok=True)\n"
            "\n"
            "    # start workers",
            "    os.makedirs(out_dir, exist_ok=True)\n"
            "    existing_samples = len(glob.glob(f'{out_dir}/sample_*.pkl'))\n"
            "    if existing_samples >= n_samples:\n"
            "        print(f'Done collecting samples for {out_dir}: already has {existing_samples}/{n_samples}')\n"
            "        return\n"
            "\n"
            "    # start workers",
        )
        text = text.replace(
            "    tmp_samples_dir = f'{out_dir}/tmp'\n"
            "    os.makedirs(tmp_samples_dir, exist_ok=True)",
            "    tmp_samples_dir = f'{out_dir}/tmp'\n"
            "    shutil.rmtree(tmp_samples_dir, ignore_errors=True)\n"
            "    os.makedirs(tmp_samples_dir, exist_ok=True)",
        )
        text = text.replace(
            "    i = 0\n"
            "    in_buffer = 0\n"
            "    while i < n_samples:",
            "    i = existing_samples\n"
            "    in_buffer = 0\n"
            "    print(f'Resuming {out_dir}: {existing_samples}/{n_samples} samples already exist')\n"
            "    while i < n_samples:",
        )

    text = text.replace("pseudo_candidates=True", "pseudo_candidates=False")
    if "pseudo_candidates=False" not in text:
        raise RuntimeError("failed to patch 02_generate_dataset.py: pseudo_candidates=False not found")
    dataset_py.write_text(text)

    text = train_py.read_text()
    if "--max-epochs" not in text:
        text = text.replace(
            "    parser.add_argument(\n"
            "        '-g', '--gpu',\n"
            "        help='CUDA GPU id (-1 for CPU).',\n"
            "        type=int,\n"
            "        default=0,\n"
            "    )",
            "    parser.add_argument(\n"
            "        '-g', '--gpu',\n"
            "        help='CUDA GPU id (-1 for CPU).',\n"
            "        type=int,\n"
            "        default=0,\n"
            "    )\n"
            "    parser.add_argument('--max-epochs', type=int, default=1000)\n"
            "    parser.add_argument('--epoch-samples', type=int, default=10000)\n"
            "    parser.add_argument('--batch-size', type=int, default=32)\n"
            "    parser.add_argument('--pretrain-batch-size', type=int, default=128)\n"
            "    parser.add_argument('--valid-batch-size', type=int, default=128)",
        )
        text = text.replace(
            "    max_epochs = 1000\n"
            "    batch_size = 32\n"
            "    pretrain_batch_size = 128\n"
            "    valid_batch_size = 128",
            "    max_epochs = args.max_epochs\n"
            "    batch_size = args.batch_size\n"
            "    pretrain_batch_size = args.pretrain_batch_size\n"
            "    valid_batch_size = args.valid_batch_size",
        )
        text = text.replace(
            "int(np.floor(10000/batch_size))*batch_size",
            "int(np.floor(args.epoch_samples/batch_size))*batch_size",
        )
    text = text.replace(
        "Scheduler(optimizer, mode='min', patience=10, factor=0.2, verbose=True)",
        "Scheduler(optimizer, mode='min', patience=10, factor=0.2)",
    )
    train_py.write_text(text)

    text = utilities_py.read_text()
    old_inc = "    def __inc__(self, key, value, store, *args, **kwargs):"
    new_inc = "    def __inc__(self, key, value, *args, **kwargs):"
    if old_inc in text:
        text = text.replace(old_inc, new_inc)
    if new_inc not in text:
        raise RuntimeError("failed to patch utilities.py: __inc__ signature not found")
    if "candidate_scores[nan_mask] = min_score - 1.0" not in text:
        old_candidate_scores = "        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])"
        if old_candidate_scores not in text:
            raise RuntimeError("failed to patch utilities.py: candidate_scores assignment not found")
        text = text.replace(
            old_candidate_scores,
            "        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])\n"
            "        nan_mask = torch.isnan(candidate_scores)\n"
            "        if nan_mask.any():\n"
            "            finite_scores = candidate_scores[torch.isfinite(candidate_scores)]\n"
            "            min_score = finite_scores.min().item() if finite_scores.numel() else 0.0\n"
            "            candidate_scores[nan_mask] = min_score - 1.0",
        )
    if "_is_better" not in text[text.find("class Scheduler") :]:
        text = text.replace(
            "        if self.is_better(current, self.best):",
            "        is_better = self.is_better if hasattr(self, 'is_better') else self._is_better\n"
            "        if is_better(current, self.best):",
        )
    utilities_py.write_text(text)

    run([sys.executable, "-m", "py_compile", str(dataset_py), str(train_py), str(utilities_py)])
    print("patches applied", flush=True)


def load_l2b_generator():
    sys.path.insert(0, str(ORIGINAL_REPO))
    spec = importlib.util.spec_from_file_location("l2b_generator", ORIGINAL_REPO / "01_generate_instances.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Learn2Branch instance generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_auction_split(generator, split_name: str, n_instances: int, n_items: int, n_bids: int, seed: int) -> None:
    import numpy as np

    out_dir = INSTANCE_DIR / f"{split_name}_{n_items}_{n_bids}"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("instance_*.lp")))
    if existing >= n_instances:
        print(f"{split_name}_{n_items}_{n_bids}: already has {existing}/{n_instances}", flush=True)
        return

    rng = np.random.RandomState(seed)
    for i in range(existing, n_instances):
        filename = out_dir / f"instance_{i + 1}.lp"
        generator.generate_cauctions(rng, str(filename), n_items=n_items, n_bids=n_bids, add_item_prob=0.7)
        if (i + 1) % 100 == 0 or (i + 1) == n_instances:
            print(f"{split_name}_{n_items}_{n_bids}: generated {i + 1}/{n_instances}", flush=True)


def generate_instances() -> None:
    generator = load_l2b_generator()

    generate_auction_split(generator, "train", N_TRAIN_INSTANCES, 100, 500, SEED)
    generate_auction_split(generator, "valid", N_VALID_INSTANCES, 100, 500, SEED + 1)
    generate_auction_split(generator, "test", N_TEST_INSTANCES, 100, 500, SEED + 2)

    generate_auction_split(generator, "transfer", N_TRANSFER_INSTANCES, 100, 500, SEED + 10)
    generate_auction_split(generator, "transfer", N_TRANSFER_INSTANCES, 200, 1000, SEED + 11)
    generate_auction_split(generator, "transfer", N_TRANSFER_INSTANCES, 300, 1500, SEED + 12)


def sample_count(split: str) -> int:
    return len(list((SAMPLE_DIR / split).glob("sample_*.pkl")))


def sample_counts() -> dict[str, int]:
    return {split: sample_count(split) for split in ("train", "valid", "test")}


def total_sample_count() -> int:
    return sum(sample_count(split) for split in ("train", "valid", "test"))


def samples_complete() -> bool:
    counts = sample_counts()
    return counts["train"] >= TRAIN_SAMPLES and counts["valid"] >= VALID_SAMPLES and counts["test"] >= TEST_SAMPLES


def collect_samples() -> None:
    if samples_complete():
        print(f"samples already complete: {sample_counts()}", flush=True)
        return

    args = [
        sys.executable,
        "02_generate_dataset.py",
        PROBLEM,
        "-j",
        str(N_JOBS),
        "--train-size",
        str(TRAIN_SAMPLES),
        "--valid-size",
        str(VALID_SAMPLES),
        "--test-size",
        str(TEST_SAMPLES),
        "--node-record-prob",
        str(NODE_RECORD_PROB),
        "--time-limit",
        str(SAMPLE_TIME_LIMIT),
    ]

    error: list[BaseException] = []

    def target() -> None:
        try:
            run(args, cwd=ECOLE_REPO)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    last_logged = (total_sample_count() // SAMPLE_LOG_EVERY) * SAMPLE_LOG_EVERY
    last_checkpoint = (total_sample_count() // SAMPLE_CHECKPOINT_EVERY) * SAMPLE_CHECKPOINT_EVERY
    last_heartbeat = time.monotonic()

    while thread.is_alive():
        count = total_sample_count()
        if count >= last_logged + SAMPLE_LOG_EVERY:
            last_logged = (count // SAMPLE_LOG_EVERY) * SAMPLE_LOG_EVERY
            print(f"[sample-progress] {count} samples written", flush=True)
        if count >= last_checkpoint + SAMPLE_CHECKPOINT_EVERY:
            last_checkpoint = (count // SAMPLE_CHECKPOINT_EVERY) * SAMPLE_CHECKPOINT_EVERY
            print(f"[sample-checkpoint] {last_checkpoint} samples visible locally; not pushing samples", flush=True)
        if time.monotonic() - last_heartbeat >= 60:
            print(f"[sample-heartbeat] running, {count} samples visible", flush=True)
            last_heartbeat = time.monotonic()
        time.sleep(5)

    thread.join()
    if error:
        raise error[0]
    print(f"[sample-progress] generator exited with {total_sample_count()} samples written", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_samples() -> None:
    if not samples_complete():
        raise RuntimeError(f"refusing to archive incomplete samples: {sample_counts()}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_archive = SAMPLE_ARCHIVE.with_name(f"{SAMPLE_ARCHIVE.name}.tmp")
    tmp_archive.unlink(missing_ok=True)

    print(f"[sample-archive] creating {SAMPLE_ARCHIVE}", flush=True)
    with tarfile.open(tmp_archive, "w:gz") as tar:
        tar.add(SAMPLE_DIR, arcname=f"data/samples/{PROBLEM_FOLDER}")
    tmp_archive.replace(SAMPLE_ARCHIVE)

    archive_hash = sha256_file(SAMPLE_ARCHIVE)
    manifest = {
        "archive": SAMPLE_ARCHIVE.name,
        "sha256": archive_hash,
        "bytes": SAMPLE_ARCHIVE.stat().st_size,
        "sample_counts": sample_counts(),
        "settings": {
            "problem": PROBLEM,
            "seed": SEED,
            "train_samples": TRAIN_SAMPLES,
            "valid_samples": VALID_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "node_record_prob": NODE_RECORD_PROB,
            "sample_time_limit": SAMPLE_TIME_LIMIT,
        },
    }
    SAMPLE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[sample-archive] wrote {SAMPLE_ARCHIVE} "
        f"({SAMPLE_ARCHIVE.stat().st_size / 1024**3:.2f}GB, sha256={archive_hash})",
        flush=True,
    )

    archive_dir = os.environ.get("SAMPLE_ARCHIVE_DIR")
    if archive_dir:
        destination = Path(archive_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        print(f"[sample-archive] copying archive and manifest to {destination}", flush=True)
        shutil.copy2(SAMPLE_ARCHIVE, destination / SAMPLE_ARCHIVE.name)
        shutil.copy2(SAMPLE_MANIFEST, destination / SAMPLE_MANIFEST.name)


def find_sample_archive_for_restore() -> Path | None:
    explicit = os.environ.get("SAMPLE_ARCHIVE_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"SAMPLE_ARCHIVE_PATH does not exist: {path}")
        return path

    archive_dir = os.environ.get("SAMPLE_ARCHIVE_DIR")
    if archive_dir:
        path = Path(archive_dir).expanduser().resolve() / SAMPLE_ARCHIVE.name
        if path.exists():
            return path
    if SAMPLE_ARCHIVE.exists():
        return SAMPLE_ARCHIVE
    return None


def restore_samples_from_archive_if_needed() -> None:
    if samples_complete():
        print(f"samples already available locally: {sample_counts()}", flush=True)
        return

    archive = find_sample_archive_for_restore()
    if archive is None:
        raise RuntimeError(
            "samples are incomplete and no archive was found. "
            "Set SAMPLE_ARCHIVE_PATH=/path/to/samples_cauctions_100_500.tar.gz "
            "or rerun L2B_STAGE=samples first."
        )

    print(f"[sample-archive] restoring samples from {archive}", flush=True)
    destination_root = ECOLE_REPO.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination_root / member.name).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tar.extractall(destination_root)
    if not samples_complete():
        raise RuntimeError(f"archive restore did not produce complete samples: {sample_counts()}")
    print(f"[sample-archive] restore complete: {sample_counts()}", flush=True)


def run_in_thread_with_monitor(args: list[str], cwd: Path, label: str) -> None:
    error: list[BaseException] = []

    def target() -> None:
        try:
            run(args, cwd=cwd)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    last_heartbeat = time.monotonic()
    while thread.is_alive():
        if time.monotonic() - last_heartbeat >= 60:
            print(f"[heartbeat] {label}: still running", flush=True)
            last_heartbeat = time.monotonic()
        time.sleep(5)
    thread.join()
    if error:
        raise error[0]


def train_model() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    gpu_id = "0" if torch.cuda.is_available() else "-1"
    args = [
        sys.executable,
        "03_train_gnn.py",
        PROBLEM,
        "-s",
        str(SEED),
        "-g",
        gpu_id,
        "--max-epochs",
        str(MAX_EPOCHS),
        "--epoch-samples",
        str(EPOCH_SAMPLES),
        "--batch-size",
        str(BATCH_SIZE),
        "--pretrain-batch-size",
        str(PRETRAIN_BATCH_SIZE),
        "--valid-batch-size",
        str(VALID_BATCH_SIZE),
    ]
    run_in_thread_with_monitor(args, ECOLE_REPO, "training")

    CHECKED_IN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_DIR / "train_params.pkl", CHECKED_IN_MODEL_DIR / "train_params.pkl")
    shutil.copy2(MODEL_DIR / "train_log.txt", CHECKED_IN_MODEL_DIR / "train_log.txt")
    git_commit_push(
        "Train auction branching model",
        [CHECKED_IN_MODEL_DIR / "train_params.pkl", CHECKED_IN_MODEL_DIR / "train_log.txt"],
    )


def tensorize_observation(observation, device: str):
    import numpy as np
    import torch

    return (
        torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
        torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
        torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
        torch.from_numpy(observation.variable_features.astype(np.float32)).to(device),
    )


def solve_internal(instance_path: Path, seed: int, time_limit: float, node_limit: int) -> dict[str, object]:
    import ecole

    scip_params = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
        "limits/nodes": int(node_limit),
        "timing/clocktype": 1,
        "branching/relpscost/priority": 9999999,
    }
    env = ecole.environment.Configuring(scip_params=scip_params)
    env.seed(seed)
    wall_start = time.perf_counter()
    env.reset(str(instance_path))
    env.step({})
    walltime = time.perf_counter() - wall_start
    model = env.model.as_pyscipopt()
    return {
        "policy": "scip:relpscost",
        "seed": seed,
        "nnodes": int(model.getNNodes()),
        "nlps": int(model.getNLPs()),
        "stime": float(model.getSolvingTime()),
        "gap": float(model.getGap()),
        "status": str(model.getStatus()),
        "walltime": float(walltime),
    }


def solve_gnn(instance_path: Path, seed: int, policy, device: str, time_limit: float, node_limit: int) -> dict[str, object]:
    import ecole
    import torch

    scip_params = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": float(time_limit),
        "limits/nodes": int(node_limit),
        "timing/clocktype": 1,
    }
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params=scip_params,
        pseudo_candidates=False,
    )
    env.seed(seed)
    torch.manual_seed(seed)
    wall_start = time.perf_counter()

    observation, action_set, _, done, _ = env.reset(str(instance_path))
    while not done:
        with torch.no_grad():
            logits = policy(*tensorize_observation(observation, device))
            action_indices = torch.as_tensor(action_set.astype("int64"), device=device)
            selected = int(torch.argmax(logits[action_indices]).item())
            action = int(action_set[selected])
        observation, action_set, _, done, _ = env.step(action)

    walltime = time.perf_counter() - wall_start
    model = env.model.as_pyscipopt()
    return {
        "policy": "gnn:supervised",
        "seed": seed,
        "nnodes": int(model.getNNodes()),
        "nlps": int(model.getNLPs()),
        "stime": float(model.getSolvingTime()),
        "gap": float(model.getGap()),
        "status": str(model.getStatus()),
        "walltime": float(walltime),
    }


def evaluate_policy() -> None:
    import torch

    sys.path.insert(0, str(ECOLE_REPO))
    from model.model import GNNPolicy

    gpu_id = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy = GNNPolicy().to(gpu_id)
    model_path = MODEL_DIR / "train_params.pkl"
    if not model_path.exists():
        model_path = CHECKED_IN_MODEL_DIR / "train_params.pkl"
    if not model_path.exists():
        raise RuntimeError("missing train_params.pkl; run L2B_STAGE=train first")
    policy.load_state_dict(torch.load(model_path, map_location=gpu_id))
    policy.eval()

    eval_instances: list[tuple[str, Path]] = []
    for size_label, folder in [
        ("small", INSTANCE_DIR / "transfer_100_500"),
        ("medium", INSTANCE_DIR / "transfer_200_1000"),
        ("big", INSTANCE_DIR / "transfer_300_1500"),
    ]:
        files = sorted(folder.glob("instance_*.lp"))[:EVAL_INSTANCES_PER_SIZE]
        for path in files:
            eval_instances.append((size_label, path))

    if not eval_instances:
        raise RuntimeError("no auction transfer instances found for evaluation")

    CHECKED_IN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = CHECKED_IN_RESULTS_DIR / "cauctions_smoke_eval.csv"
    fieldnames = [
        "size",
        "instance",
        "policy",
        "seed",
        "nnodes",
        "nlps",
        "stime",
        "gap",
        "status",
        "walltime",
    ]
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for size_label, instance in eval_instances:
            print(f"[eval] {size_label}: {instance}", flush=True)
            for seed in range(EVAL_SEEDS):
                scip = solve_internal(instance, seed, EVAL_TIME_LIMIT, EVAL_NODE_LIMIT)
                writer.writerow({"size": size_label, "instance": str(instance), **scip})
                handle.flush()
                print(
                    f"  scip seed={seed}: {scip['nnodes']} nodes "
                    f"{scip['walltime']:.2f}s {scip['status']}",
                    flush=True,
                )

                gnn = solve_gnn(instance, seed, policy, gpu_id, EVAL_TIME_LIMIT, EVAL_NODE_LIMIT)
                writer.writerow({"size": size_label, "instance": str(instance), **gnn})
                handle.flush()
                print(
                    f"  gnn seed={seed}: {gnn['nnodes']} nodes "
                    f"{gnn['walltime']:.2f}s {gnn['status']}",
                    flush=True,
                )

    git_commit_push("Evaluate auction branching model", [result_path])


def main() -> None:
    stage = os.environ.get("L2B_STAGE", "all").lower()
    valid_stages = {"all", "samples", "train", "evaluate"}
    if stage not in valid_stages:
        raise RuntimeError(f"L2B_STAGE must be one of {sorted(valid_stages)}, got {stage!r}")

    require_python_311()
    print_system_info()
    pip_install_missing()

    import ecole
    import torch
    import torch_geometric

    print(f"ecole: {ecole.__version__}", flush=True)
    print(f"torch_geometric: {torch_geometric.__version__}", flush=True)
    print(f"cuda available after setup: {torch.cuda.is_available()}", flush=True)

    ensure_repos()
    patch_scripts()

    if stage in {"all", "samples"}:
        generate_instances()
        collect_samples()
        archive_samples()
        if stage == "samples":
            print("finished after sample generation/archive", flush=True)
            return

    if stage in {"all", "train"}:
        restore_samples_from_archive_if_needed()
        train_model()
        if stage == "train":
            print("finished after training", flush=True)
            return

    if stage in {"all", "evaluate"}:
        generate_instances()
        evaluate_policy()

    print("vast cauctions runner finished", flush=True)


if __name__ == "__main__":
    main()
