"""Runs inside a Kaggle kernel (pushed by kagglejobs/run.py).

The repository arrives as the private dataset ``<user>/jex-src`` (a tarball of the tracked files), so no
GitHub token is needed on Kaggle. ``CONFIG`` is injected by run.py at the top of the pushed copy; the
steps write their results to /kaggle/working/results, which ``kaggle kernels output`` downloads.

Runs locally too (for a dry check): JEX_INPUT_DIR, JEX_WORK_DIR and JEX_OUT_DIR override the Kaggle paths.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

CONFIG = globals().get("CONFIG") or json.loads(os.environ.get("JEX_JOB_CONFIG", "{}"))

INPUT = Path(os.environ.get("JEX_INPUT_DIR", "/kaggle/input"))
WORK = Path(os.environ.get("JEX_WORK_DIR", "/kaggle/tmp"))  # not saved as output
OUT = Path(os.environ.get("JEX_OUT_DIR", "/kaggle/working"))
RESULTS = OUT / "results"
REPO = WORK / "jex"
HARNESS = WORK / "jev-benchmarking"
ZENODO = "https://zenodo.org/api/records/23039006/files"
REFERENCE_DBS = ["responses.db", "responses.Qwen__Qwen3.8-27B.db", "responses.google__gemma-4-E4B-it.db"]

failed: list[str] = []


def sh(cmd: str, cwd: Path | None = None, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=cwd or REPO).returncode
    if rc and check:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


def step(name):
    def wrap(fn):
        def run():
            t0 = time.time()
            print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
            try:
                fn()
                print(f"[{name}] ok in {time.time() - t0:.0f}s", flush=True)
            except Exception as e:  # keep going: later steps and the bundle still run
                failed.append(name)
                print(f"[{name}] FAILED after {time.time() - t0:.0f}s: {e}", flush=True)

        return run

    return wrap


def find_source() -> Path:
    """The dataset may arrive as the tarball or already extracted, depending on Kaggle's handling."""
    tars = glob.glob(str(INPUT / "**" / "jex-src.tar.gz"), recursive=True)
    if tars:
        REPO.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tars[0]) as t:
            t.extractall(REPO)
        return REPO
    for p in glob.glob(str(INPUT / "**" / "jex" / "backbone.py"), recursive=True):
        src = Path(p).parent.parent
        shutil.copytree(src, REPO, dirs_exist_ok=True)
        return REPO
    raise FileNotFoundError(f"jex source not found under {INPUT}")


@step("setup")
def setup():
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    find_source()
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true", check=False)
    sh(f"{sys.executable} -m pip install -q -e '.[server,dev,gpu]'")
    # the image ships torchao 0.10; peft refuses to create LoRA layers next to a torchao < 0.16, jex does not use it
    sh(f"{sys.executable} -m pip uninstall -q -y torchao", check=False)
    if CONFIG.get("fla", True):
        sh(f"{sys.executable} -m pip install -q flash-linear-attention", check=False)
    sh(f"{sys.executable} -c \"import torch, transformers; print('torch', torch.__version__, 'cuda', "
       f"torch.cuda.is_available(), 'transformers', transformers.__version__)\"")


@step("tests")
def tests():
    sh(f"{sys.executable} -m pytest -q -x")


@step("pipeline")
def pipeline():
    c = CONFIG
    q = f"--teacher-quant {c['teacher_quant']}" if c.get("teacher_quant") else ""
    s = c.get("sizes", {})
    sh(f"{sys.executable} scripts/extract.py --out {WORK}/feats --student {c['student']} --teacher {c['teacher']} {q} "
       f"--train-per-task {s.get('train', 400)} --dev-per-task {s.get('dev', 50)} "
       f"--eval-per-task {s.get('eval', 150)} --heldout-per-task {s.get('heldout', 400)}")
    heads = []
    for name, flags in c.get("heads", {"jex-head": ""}).items():
        sh(f"{sys.executable} scripts/train_head.py --feats {WORK}/feats --out {WORK}/{name} {flags}", check=False)
        heads.append(f"{WORK}/{name}")
    sh(f"{sys.executable} scripts/evaluate.py --feats {WORK}/feats --heads {' '.join(heads)} --out {RESULTS}/eval.json")
    for h in heads:
        shutil.copy(Path(h) / "train_info.json", RESULTS / f"train_{Path(h).name}.json")
    shutil.copy(Path(WORK) / "feats" / "meta.json", RESULTS / "feats_meta.json")


@step("lora")
def lora():
    c = CONFIG
    sh(f"{sys.executable} scripts/train_lora.py --feats {WORK}/feats --out {OUT}/lora "
       f"--max-records {c.get('lora_max_records', 3000)}")
    sh(f"{sys.executable} scripts/extract.py --student {c['student']} --student-adapter {OUT}/lora "
       f"--records-from {WORK}/feats/records.pt --splits dev,eval --skip-teacher --out {WORK}/feats-lora")
    sh(f"{sys.executable} scripts/evaluate.py --feats {WORK}/feats-lora --heads --out {RESULTS}/eval_lora.json")


@step("jev_bench")
def jev_bench():
    if not HARNESS.exists():
        sh(f"git clone -q https://github.com/AppliedMachineLearning-Lab/jev-benchmarking {HARNESS}", cwd=WORK)
    sh(f"{sys.executable} -m pip install -q -e {HARNESS}")
    (HARNESS / "cache").mkdir(exist_ok=True)
    for db in REFERENCE_DBS:
        if not (HARNESS / "cache" / db).exists():
            sh(f"curl -sSL -o {db} {ZENODO}/{db}/content", cwd=HARNESS / "cache")
    limit = CONFIG.get("bench_limit", 30)
    tasks = " ".join(CONFIG.get("bench_tasks", ["all"]))
    labels = []
    for tag, (model, quant) in CONFIG.get("bench_models", {}).items():
        # "@lora" = a checkpoint produced by an earlier step of this job (in /kaggle/working)
        src = f"--checkpoint {OUT / model[1:]}" if model.startswith("@") else f"--backbone {model}"
        qflag = f"--quant {quant}" if quant else ""
        rc = sh(f"{sys.executable} scripts/jev_bench.py {tasks} {src} {qflag} --tag {tag} --limit {limit} "
                f"--out {WORK}/runs/{tag}", check=False)
        if rc:
            failed.append(f"jev_bench:{tag}")
            continue
        sh(f"{sys.executable} scripts/import_responses.py {WORK}/runs/{tag}/*.jsonl", cwd=HARNESS)
        sh(f"{sys.executable} scripts/evaluate.py {tasks} --split eval --limit {limit} --no-ci --model hf:jex/{tag}",
           cwd=HARNESS, check=False)
        labels.append(f'--model "{tag}=hf:jex/{tag}"')
    for ref in ["jev-1.13.0", "hf:Qwen/Qwen3.8-27B", "hf:google/gemma-4-E4B-it"]:
        sh(f"{sys.executable} scripts/evaluate.py {tasks} --split eval --limit {limit} --no-ci --model {ref}",
           cwd=HARNESS, check=False)
    sh(f"{sys.executable} scripts/compare_jev.py --harness {HARNESS} --model Jev=jev-1.13.0 "
       f"--model 'Qwen3.8-27B=hf:Qwen/Qwen3.8-27B' --model 'Gemma-4-E4B=hf:google/gemma-4-E4B-it' "
       f"{' '.join(labels)} --out {RESULTS}/jev_compare.md")
    shutil.copytree(HARNESS / "results" / "eval" / "open_models", RESULTS / "jev_open_models", dirs_exist_ok=True)


@step("latency")
def latency():
    sh(f"{sys.executable} scripts/bench_latency.py --backbone {CONFIG['student']} --ks 1 2 4 8 16 --repeats 5 "
       f"--out {RESULTS}/bench_latency.json")


@step("async")
def async_bench():
    sh(f"{sys.executable} scripts/bench_async.py --backbone {CONFIG['student']} --rates 5 20 50 --n 200 "
       f"--out {RESULTS}/bench_async.json")


STEPS = {"setup": setup, "tests": tests, "pipeline": pipeline, "lora": lora, "jev_bench": jev_bench,
         "latency": latency, "async": async_bench}


def main():
    print("CONFIG", json.dumps(CONFIG, indent=1), flush=True)
    t0 = time.time()
    for name in CONFIG.get("steps", ["setup", "tests"]):
        STEPS[name]()
        if name == "setup" and "setup" in failed:
            break
    summary = {"config": CONFIG, "failed": failed, "seconds": round(time.time() - t0)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nSUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
