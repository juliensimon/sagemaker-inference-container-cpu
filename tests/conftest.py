"""
Pytest configuration and fixtures for the SageMaker inference container test suite.

This module provides shared fixtures, test configuration, and utilities
for all test modules in the project.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide a directory for test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for each test."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_env() -> Generator[Dict[str, str], None, None]:
    """Provide a clean environment for each test."""
    original_env = os.environ.copy()
    os.environ.clear()
    yield {}
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def fastapi_client() -> TestClient:
    """Provide a FastAPI test client."""
    return TestClient(app)


@pytest.fixture(scope="function")
def mock_httpx_client() -> Generator[Mock, None, None]:
    """Provide a mocked httpx client for testing HTTP requests."""
    with patch("httpx.AsyncClient") as mock_client:
        yield mock_client


@pytest.fixture(scope="function")
def mock_subprocess() -> Generator[Mock, None, None]:
    """Provide a mocked subprocess for testing process spawning."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdout = Mock()
        mock_process.stdout.readline.return_value = b"test output\n"
        mock_popen.return_value = mock_process
        yield mock_popen


@pytest.fixture(scope="function")
def mock_psutil() -> Generator[Mock, None, None]:
    """Provide a mocked psutil for testing process management."""
    with patch("psutil.Process") as mock_process_class:
        mock_process = Mock()
        mock_process.children.return_value = []
        mock_process.terminate.return_value = None
        mock_process.wait.return_value = None
        mock_process_class.return_value = mock_process
        yield mock_process_class


@pytest.fixture(scope="function")
def mock_boto3() -> Generator[Mock, None, None]:
    """Provide a mocked boto3 client for testing S3 operations."""
    with patch("boto3.client") as mock_client:
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": [{"Key": "test/model.gguf"}]}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_client.return_value = mock_s3
        yield mock_client


@pytest.fixture(scope="function")
def mock_huggingface_hub() -> Generator[Mock, None, None]:
    """Provide a mocked huggingface_hub for testing HF downloads."""
    with patch("huggingface_hub.snapshot_download") as mock_snapshot, patch(
        "huggingface_hub.hf_hub_download"
    ) as mock_download:
        mock_snapshot.return_value = None
        mock_download.return_value = "/tmp/test/model.gguf"
        yield {"snapshot": mock_snapshot, "download": mock_download}


@pytest.fixture(scope="function")
def sample_gguf_model(temp_dir: Path) -> Path:
    """Provide a sample GGUF model file for testing."""
    model_path = temp_dir / "test_model.gguf"
    model_path.write_bytes(b"fake gguf model content")
    return model_path


@pytest.fixture(scope="function")
def sample_hf_model(temp_dir: Path) -> Path:
    """Provide a sample HuggingFace model directory for testing."""
    model_dir = temp_dir / "hf_model"
    model_dir.mkdir()

    # Create config.json
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
    }
    (model_dir / "config.json").write_text(str(config))

    # Create a sample safetensors file
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(
        b"fake safetensors content"
    )

    return model_dir


@pytest.fixture(scope="function")
def mock_llama_server_binary(temp_dir: Path) -> Path:
    """Provide a mock llama-server binary for testing."""
    binary_path = temp_dir / "llama-server"
    binary_path.write_text("#!/bin/bash\necho 'mock llama-server'")
    binary_path.chmod(0o755)
    return binary_path


@pytest.fixture(scope="function")
def mock_quantize_binary(temp_dir: Path) -> Path:
    """Provide a mock llama-quantize binary for testing."""
    binary_path = temp_dir / "llama-quantize"
    binary_path.write_text("#!/bin/bash\necho 'mock llama-quantize'")
    binary_path.chmod(0o755)
    return binary_path


@pytest.fixture(scope="function")
def mock_convert_script(temp_dir: Path) -> Path:
    """Provide a mock convert_hf_to_gguf.py script for testing."""
    script_path = temp_dir / "convert_hf_to_gguf.py"
    script_path.write_text("#!/usr/bin/env python3\nprint('mock conversion script')")
    script_path.chmod(0o755)
    return script_path


@pytest.fixture(scope="function")
def mock_models_dir(temp_dir) -> Generator[Path, None, None]:
    """Provide a mocked MODELS_DIR to prevent permission errors."""
    with patch("app.model_manager.MODELS_DIR", temp_dir / "models"):
        yield temp_dir / "models"


@pytest.fixture(scope="function")
def mock_llamacpp_dir(temp_dir) -> Generator[Path, None, None]:
    """Provide a mocked LLAMACPP_DIR to prevent permission errors."""
    with patch("app.model_manager.LLAMACPP_DIR", temp_dir / "llamacpp"):
        yield temp_dir / "llamacpp"


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line(
        "markers", "container: mark test as container integration test"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "env: mark test as environment variable test")
    config.addinivalue_line("markers", "mock: mark test as using mocks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add unit marker to tests that don't have integration marker
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)

        # Add mock marker to tests that use mocks
        if any("mock" in str(arg) for arg in item.funcargs.values()):
            item.add_marker(pytest.mark.mock)
