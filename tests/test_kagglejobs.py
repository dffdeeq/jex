import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("kaggle_run", ROOT / "kagglejobs" / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


@pytest.mark.parametrize("job", sorted(run.PRESETS))
def test_injected_config_is_python(job, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "BUILD", tmp_path)
    config = {**run.BASE, **run.PRESETS[job]}
    src = (run.write_kernel("user", job, config) / "job.py").read_text()
    first = ast.parse(src).body[0]  # CONFIG = {...}; None/True, not null/true
    assert eval(compile(ast.Expression(first.value), "job.py", "eval"), {}) == config
