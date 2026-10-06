"""Run jex jobs on Kaggle GPUs from the command line.

    export KAGGLE_API_TOKEN=...            # kaggle.com/settings/api -> "Generate New Token"
    python kagglejobs/run.py smoke         # ~10 min: GPU check, tests, tiny Jev-benchmark run
    python kagglejobs/run.py jev-bench     # Jev benchmark for the models in PRESETS["jev-bench"]
    python kagglejobs/run.py full          # data -> heads -> LoRA -> eval -> Jev bench -> latency/async
    python kagglejobs/run.py status full   # status of the last run
    python kagglejobs/run.py output full   # download results again

What `run` does: tars the tracked repository files into the private dataset <user>/jex-src (new version on
every run), writes a kernel folder with kernel-metadata.json and job.py (CONFIG injected), pushes it to a T4
machine with internet, waits for it and downloads /kaggle/working into runs/kaggle/<job>/.
"""

from __future__ import annotations

import argparse
import json
import pprint
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "kaggle"
RUNS = ROOT / "runs" / "kaggle"
SRC_SLUG = "jex-src"

BASE = {"student": "Qwen/Qwen2.5-1.5B-Instruct", "teacher": "Qwen/Qwen2.5-7B-Instruct", "teacher_quant": "4bit"}
PRESETS = {
    "smoke": {
        "steps": ["setup", "tests", "jev_bench"],
        "bench_models": {"qwen2.5-0.5b-zs": ["Qwen/Qwen2.5-0.5B-Instruct", None]},
        "bench_tasks": ["sst2", "banking77", "go_emotions"],
        "bench_limit": 5,
        "timeout": 3600,
    },
    "jev-bench": {
        "steps": ["setup", "jev_bench"],
        "bench_models": {
            "qwen2.5-1.5b-zs": ["Qwen/Qwen2.5-1.5B-Instruct", None],
            "qwen3.5-4b-zs": ["Qwen/Qwen3.5-4B", None],
            "qwen3.5-9b-zs": ["Qwen/Qwen3.5-9B", "4bit"],
        },
        "bench_limit": 30,
        "timeout": 6 * 3600,
    },
    "full": {
        "steps": ["setup", "tests", "pipeline", "lora", "jev_bench", "latency", "async"],
        "sizes": {"train": 400, "dev": 50, "eval": 150, "heldout": 400},
        "heads": {"jex-head": "", "ablation-gold-only": "--no-teacher", "prior-kl-1.0": "--no-memory --prior-kl 1.0"},
        "lora_max_records": 3000,
        "bench_models": {"jex-lora": ["@lora", None], "qwen2.5-1.5b-zs": ["Qwen/Qwen2.5-1.5B-Instruct", None]},
        "bench_limit": 30,
        "timeout": 11 * 3600,
    },
}


def kaggle(*args: str, capture: bool = False) -> str:
    cmd = ["kaggle", *args]
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, text=True, capture_output=capture)
    if res.returncode:
        if capture:
            print(res.stdout, res.stderr, file=sys.stderr)
        raise SystemExit(f"kaggle {' '.join(args)} failed ({res.returncode})")
    return res.stdout if capture else ""


def username() -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.config_values.get("username")
    if not user:
        raise SystemExit("could not determine the Kaggle username; set KAGGLE_USERNAME")
    return user


def push_source(user: str) -> None:
    """Tar the git-tracked files (the working tree versions) into the private dataset <user>/jex-src."""
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.split()
    folder = BUILD / SRC_SLUG
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(folder / "jex-src.tar.gz", "w:gz") as tar:
        for f in files:
            if (ROOT / f).is_file():
                tar.add(ROOT / f, arcname=f)
    meta = {"title": SRC_SLUG, "id": f"{user}/{SRC_SLUG}", "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SRC_SLUG}"], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        kaggle("datasets", "version", "-p", str(folder), "-m", f"jex {rev}", "-q")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-q")
    # the new version has to be processed before a kernel can mount it
    for _ in range(60):
        out = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SRC_SLUG}"], capture_output=True, text=True)
        if "ready" in out.stdout.lower():
            return
        time.sleep(10)
    raise SystemExit("dataset jex-src did not become ready in 10 minutes")


def write_kernel(user: str, job: str, config: dict) -> Path:
    slug = f"jex-{job}"
    folder = BUILD / slug
    folder.mkdir(parents=True, exist_ok=True)
    body = (ROOT / "kagglejobs" / "job.py").read_text()
    # a Python literal, not JSON: null/true/false would be NameErrors in the kernel
    (folder / "job.py").write_text(f"CONFIG = {pprint.pformat(config, sort_dicts=False)}\n\n{body}")
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": config.get("machine_shape", "NvidiaTeslaT4"),
        "dataset_sources": [f"{user}/{SRC_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return folder


def wait(ref: str, poll: int = 60) -> str:
    while True:
        out = kaggle("kernels", "status", ref, capture=True).strip()
        print(time.strftime("%H:%M:%S"), out, flush=True)
        low = out.lower()
        if any(s in low for s in ("complete", "error", "cancel")):
            return low
        time.sleep(poll)


def download(ref: str, job: str) -> Path:
    dest = RUNS / job
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", str(dest), "-o", "-q")
    summary = dest / "results" / "summary.json"
    if summary.exists():
        print(summary.read_text())
    compare = dest / "results" / "jev_compare.md"
    if compare.exists():
        print(compare.read_text())
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", help="a preset to run (smoke, jev-bench, full), or status / output")
    ap.add_argument("job", nargs="?", help="preset name for status / output")
    ap.add_argument("--set", action="append", default=[], help="override a CONFIG key: --set bench_limit=10")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only write build/kaggle/*, push nothing")
    args = ap.parse_args()

    if args.command in ("status", "output"):
        user = username()
        ref = f"{user}/jex-{args.job}"
        if args.command == "status":
            print(kaggle("kernels", "status", ref, capture=True))
        else:
            download(ref, args.job)
        return

    job = args.command
    if job not in PRESETS:
        raise SystemExit(f"unknown preset {job!r}; one of {sorted(PRESETS)}")
    config = {**BASE, **PRESETS[job]}
    for kv in args.set:
        key, value = kv.split("=", 1)
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
    timeout = str(config.pop("timeout", 6 * 3600))

    user = "DRY-RUN-USER" if args.dry_run else username()
    folder = write_kernel(user, job, config)
    if args.dry_run:
        print(f"wrote {folder}")
        return
    push_source(user)
    kaggle("kernels", "push", "-p", str(folder), "-t", timeout)
    ref = f"{user}/jex-{job}"
    print(f"https://www.kaggle.com/code/{ref}")
    if args.no_wait:
        return
    state = wait(ref)
    download(ref, job)
    if "complete" not in state:
        raise SystemExit(f"kernel finished with status: {state}")


if __name__ == "__main__":
    main()
