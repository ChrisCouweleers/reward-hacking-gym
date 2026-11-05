"""Run untrusted, model-written Python in a resource-limited subprocess.

This is a *research* sandbox, not a security boundary. It stops the common
failure modes of RL-generated code (infinite loops, memory blowups, huge
writes, stray files) from taking down a training run. Run training inside a
disposable container (e.g. a Kaggle/Colab VM) for real isolation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:  # resource is POSIX-only
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None


@dataclass
class RunResult:
    returncode: int | None  # None means the process was killed by the timeout
    stdout: str
    stderr: str
    timed_out: bool


def _limit_resources(mem_mb: int, cpu_s: int, fsize_mb: int):
    def apply():
        if resource is None:
            return
        mem = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        fsize = fsize_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        os.setsid()  # own process group so the timeout can kill children too

    return apply


def run_python(
    files: dict[str, str],
    main: str,
    args: list[str] | None = None,
    timeout_s: float = 5.0,
    mem_mb: int = 512,
    max_output_chars: int = 20_000,
) -> RunResult:
    """Write ``files`` into a fresh temp dir and run ``python -I main *args`` there."""
    with tempfile.TemporaryDirectory(prefix="rhgym_") as workdir:
        for name, content in files.items():
            with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
                f.write(content)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [sys.executable, "-I", main, *(args or [])]
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            preexec_fn=_limit_resources(mem_mb, int(timeout_s) + 1, 10) if resource else None,
        )
        try:
            out, err = proc.communicate(timeout=timeout_s)
            return RunResult(proc.returncode, out[-max_output_chars:], err[-max_output_chars:], False)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, _ = proc.communicate()
            return RunResult(None, (out or "")[-max_output_chars:], "TIMEOUT", True)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()
