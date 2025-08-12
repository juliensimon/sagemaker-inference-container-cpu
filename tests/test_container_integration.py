import os
import shutil
import socket
import subprocess
import time
from typing import Optional

import pytest
import requests


def _require_docker() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI not available on PATH; skipping container tests")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(cmd: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=True,
    )


@pytest.mark.integration
@pytest.mark.container
def test_container_builds_successfully() -> None:
    """Build the container and assert success."""
    _require_docker()
    tag = "sagemaker-llamacpp-graviton:test"
    # Build
    result = _run(["docker", "build", "-t", tag, "."], timeout=60 * 30)
    assert result.returncode == 0
    assert (
        f"Successfully tagged {tag}" in result.stdout
        or "writing image sha256:" in result.stdout
    )

    # Inspect
    inspect = _run(["docker", "image", "inspect", tag])
    assert inspect.returncode == 0

    # Cleanup image to keep CI light (leave it if user wants caching)
    _run(["docker", "rmi", "-f", tag])


@pytest.mark.integration
@pytest.mark.container
@pytest.mark.slow
def test_container_runs_with_real_model() -> None:
    """Run the built container with a real HF model and hit health + models endpoints.

    This test is opt-in to avoid long downloads by default. Enable with RUN_REAL_MODEL_TESTS=1.
    You can override default model via REAL_HF_MODEL_ID and REAL_MODEL_FILENAME.
    Optionally pass HF_TOKEN for gated models.
    """
    if os.getenv("RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Set RUN_REAL_MODEL_TESTS=1 to run real-model container test")

    _require_docker()

    # Defaults picked for relatively small GGUF availability; override if desired
    model_id = os.getenv("REAL_HF_MODEL_ID", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    model_filename = os.getenv(
        "REAL_MODEL_FILENAME", "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    )
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    tag = "sagemaker-llamacpp-graviton:run"

    # Build (reuse cache if present)
    _run(["docker", "build", "-t", tag, "."], timeout=60 * 30)

    port = _free_port()
    envs = ["-e", f"HF_MODEL_ID={model_id}", "-e", f"MODEL_FILENAME={model_filename}"]
    if hf_token:
        envs += ["-e", f"HF_TOKEN={hf_token}"]

    # Run detached
    run_cmd = [
        "docker",
        "run",
        "-d",
        "-p",
        f"{port}:8080",
        *envs,
        "--name",
        f"llamacpp_test_{port}",
        tag,
    ]
    container_id = _run(run_cmd, timeout=60 * 5).stdout.strip()
    try:
        # Wait for /ping
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + int(
            os.getenv("MAX_STARTUP_SECS", "900")
        )  # up to 15 min for first-time download
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(f"{base}/ping", timeout=5)
                if r.status_code == 200 and r.text == "OK":
                    break
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(5)
        else:
            pytest.fail(
                f"Container did not become healthy in time; last_err={last_err}"
            )

        # Verify models endpoint
        r = requests.get(f"{base}/v1/models", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        # OpenAI schema: {"object":"list","data":[{"id":...}]}
        if "data" in data and isinstance(data["data"], list) and data["data"]:
            assert "id" in data["data"][0]

    finally:
        # Cleanup container
        try:
            _run(["docker", "rm", "-f", container_id])
        except Exception:
            pass
        # Optionally remove image
        try:
            _run(["docker", "rmi", "-f", tag])
        except Exception:
            pass
