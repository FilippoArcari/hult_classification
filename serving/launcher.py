"""Start the existing Next.js UI and its upload-only API with one command."""

import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

from hydra.utils import to_absolute_path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def wait_for_url(url, processes, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"Page service exited with code {process.returncode}")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {url}")


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_page(cfg):
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("Node.js/npm is required to start frontend/.")
    for port in (8000, cfg.port):
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already in use. Stop its service and retry.")
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules" / "next" / "package.json").exists():
        log.info("Installing frontend dependencies (first launch only)")
        subprocess.run([npm, "ci"], cwd=frontend, check=True)
    env = os.environ.copy()
    env["MODEL_CHECKPOINT_PATH"] = to_absolute_path(cfg.checkpoint)
    env["CLASS_NAMES_PATH"] = to_absolute_path(cfg.class_names)
    processes = []
    try:
        processes.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "serving.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT, env=env, start_new_session=os.name == "posix",
        ))
        processes.append(subprocess.Popen(
            [npm, "run", "dev", "--", "--hostname", "127.0.0.1", "--port", str(cfg.port)],
            cwd=frontend, env=env, start_new_session=os.name == "posix",
        ))
        url = f"http://127.0.0.1:{cfg.port}"
        wait_for_url("http://127.0.0.1:8000/health", processes)
        wait_for_url(url, processes)
        log.info("UI ready: %s (Ctrl+C to stop both services)", url)
        if cfg.open_browser:
            webbrowser.open(url)
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("A page service stopped; shutting down the other service.")
    except KeyboardInterrupt:
        log.info("Stopping page services")
    finally:
        for process in reversed(processes):
            stop_process(process)
