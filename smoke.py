"""QA for the modded-nanogpt speedrun on one H100: the candidate's train_gpt.py compiles and trains.

The agent's sandbox has no GPU, so code that only fails on one (an Inductor error while compiling, a CUDA
kernel, a shape mismatch in the compiled graph) would otherwise first show in the 8×H100 benchmark, where a
crash can't be sent back to the agent. This runs the record folder the candidate changed, as speedrun-8xh100
will run it (same packages, CUDA headers, hub kernels and data path), with `torchrun --nproc_per_node=1`, until
SMOKE_STEPS training steps have printed, then stops it. A run that fails first is sent back with its first
traceback. It says nothing about speed or loss; the benchmark measures those.

Environment variables: SMOKE_STEPS (20), SMOKE_TIMEOUT_S (1800, compiling on one GPU takes several minutes).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

RESULT_FILE = os.environ.pop("HARNESS_RESULT_FILE", None)  # before anything runs the code under test


def report(result: dict) -> None:
    """Print the result, and write it to the file Kupanga reads it from (docs/evals.md)."""
    line = json.dumps(result)
    print(line, flush=True)
    if RESULT_FILE:
        with open(RESULT_FILE, "w") as f:
            f.write(line + "\n")


import bench  # noqa: E402  (speedrun-8xh100's bench.py, shipped with this eval: the same setup it uses)

SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "20"))
SMOKE_TIMEOUT_S = int(os.environ.get("SMOKE_TIMEOUT_S", "1800"))
TRAIN_STEP = re.compile(r"^step:(\d+)/(\d+) train_time:", re.M)
SHARDS = ["fineweb_val_000000.bin", "fineweb_train_000001.bin", "fineweb_train_000002.bin"]  # the loader opens a second


def main() -> None:
    subject = bench.find_subject()
    rel = bench.record_dir(subject)
    print(f"subject: {subject / rel}", flush=True)
    with tempfile.TemporaryDirectory(prefix="speedrun-smoke-") as tmp:
        tmp = Path(tmp)
        d = tmp / "dataroot" / "data" / "fineweb10B"
        d.mkdir(parents=True)
        for name in SHARDS:
            bench.fetch(name, d)
        os.environ.setdefault("CUDA_HOME", bench.cuda_home(tmp / "cuda"))
        code = tmp / "work" / subject.name
        shutil.copytree(subject, code, symlinks=True, ignore=shutil.ignore_patterns("logs", "__pycache__"))
        code = code / rel
        bench.prefetch_kernels(code)
        out = tmp / "smoke.out"
        env = {**os.environ, "DATA_PATH": str(tmp / "dataroot"), "PYTHONUNBUFFERED": "1"}
        t0 = time.time()
        with open(out, "w") as f:
            p = subprocess.Popen(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=code, env=env,
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            reached, first_step_s = 0, None
            while p.poll() is None and time.time() - t0 < SMOKE_TIMEOUT_S:
                time.sleep(5)
                steps = [int(m.group(1)) for m in TRAIN_STEP.finditer(out.read_text(errors="replace"))]
                if steps and first_step_s is None:
                    first_step_s = time.time() - t0
                reached = max(steps, default=0)
                if reached >= SMOKE_STEPS:
                    break
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        text = out.read_text(errors="replace")
    wall = time.time() - t0
    problems = []
    if reached < SMOKE_STEPS:
        why = (f"stopped after {SMOKE_TIMEOUT_S} s without reaching training step {SMOKE_STEPS}" if p.returncode == -9
               else f"exited with code {p.returncode} at training step {reached}")
        msg = f"On one H100, {rel}/train_gpt.py {why}. {bench.failure_excerpt(text)}"
        problems.append(msg if len(msg) <= 8000 else msg[:3000] + "\n...\n" + msg[-5000:])  # the error is at the end
    print(f"reached step {reached} in {wall:.0f}s (first step at {first_step_s or 0:.0f}s)", flush=True)
    report({"problems": len(problems), "critique": problems, "steps_reached": reached,
            "first_step_s": round(first_step_s or -1.0, 1), "wall_s": round(wall, 1)})


if __name__ == "__main__":
    main()
