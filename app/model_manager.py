"""
Model management utilities for SageMaker inference container.

This module handles model preparation, downloading, conversion, and quantization.
It supports downloading models from HuggingFace Hub or S3, converting HuggingFace
models to GGUF format, and quantizing models for optimal performance.

The module provides a unified interface for model preparation that automatically
handles different model sources and formats, ensuring the llama.cpp server has
access to a properly formatted GGUF model.

Environment Variables:
    MODELS_DIR: Directory to store models (default: /opt/models)
    LLAMACPP_DIR: Directory containing llama.cpp binaries (default: /opt/llama.cpp)
    MODEL_FILENAME: Specific GGUF model file to use (must be a .gguf file)
    QUANTIZATION: Quantization type (e.g., q4_k_m, q8_0)
    HF_MODEL_ID: HuggingFace model ID for downloading
    HF_MODEL_URI: S3 URI for downloading models
"""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from app.sources_hf import download_hf
from app.sources_s3 import download_s3

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/models"))
LLAMACPP_DIR = Path(os.getenv("LLAMACPP_DIR", "/opt/llama.cpp"))


def _find_quantize_binary() -> Optional[str]:
    """
    Find the llama-quantize binary in the system PATH.

    Returns:
        Full path to the quantize binary, or None if not found
    """
    return shutil.which("llama-quantize")


def _find_convert_script() -> Optional[Path]:
    """
    Find the HuggingFace to GGUF conversion script.

    Returns:
        Path to the conversion script, or None if not found
    """
    script = LLAMACPP_DIR / "convert_hf_to_gguf.py"
    return script if script.exists() else None


def _convert_hf_to_gguf(source_dir: Path, out_dir: Path) -> Path:
    """
    Convert a HuggingFace model to GGUF format.

    Args:
        source_dir: Directory containing the HuggingFace model
        out_dir: Output directory for the GGUF file

    Returns:
        Path to the converted GGUF file

    Raises:
        RuntimeError: If conversion script is not found or conversion fails
    """
    script = _find_convert_script()
    if not script:
        raise RuntimeError(f"convert_hf_to_gguf.py not found in {LLAMACPP_DIR}")

    print(f"Using conversion script: {script}")

    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "model-f16.gguf"
    cmd = [
        "/opt/venv/bin/python3",
        str(script),
        "--outtype",
        "f16",
        "--outfile",
        str(outfile),
        str(source_dir),
    ]

    print(f"Running conversion command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return outfile


def _quantize_gguf(src_path: Path, qtype: str) -> Path:
    """
    Quantize a GGUF model to a specific quantization type.

    Args:
        src_path: Path to the source GGUF file
        qtype: Quantization type (e.g., q4_k_m, q8_0)

    Returns:
        Path to the quantized GGUF file

    Raises:
        RuntimeError: If quantize binary is not found or quantization fails
    """
    quant_bin = _find_quantize_binary()
    if not quant_bin:
        raise RuntimeError("llama-quantize binary not found")

    print(f"Using quantize binary: {quant_bin}")

    stem = src_path.name[: -len(src_path.suffix)]
    out_path = src_path.parent / f"{stem}.{qtype}.gguf"

    cmd = [quant_bin, str(src_path), str(out_path), qtype]
    print(f"Running quantization command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


def _looks_like_hf_repo(dir_path: Path) -> bool:
    """
    Check if a directory looks like a HuggingFace model repository.

    Args:
        dir_path: Directory to check

    Returns:
        True if the directory contains HuggingFace model files
    """
    return (dir_path / "config.json").exists() or any(dir_path.glob("*.safetensors"))


def _detect_model_type_from_s3_uri(s3_uri: str) -> str:
    """
    Detect if an S3 URI points to GGUF or safetensors files.

    Args:
        s3_uri: S3 URI to analyze

    Returns:
        'gguf' if URI points to GGUF files, 'safetensors' if safetensors, 'unknown' otherwise
    """
    import boto3

    from app.sources_s3 import _parse_s3_uri

    try:
        s3 = boto3.client("s3")
        bucket, prefix = _parse_s3_uri(s3_uri)

        # Check a few files to determine type
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".gguf"):
                    return "gguf"
                elif key.endswith(".safetensors"):
                    return "safetensors"
        return "unknown"
    except Exception:
        return "unknown"


def prepare_model_and_get_path() -> str:
    """
    Prepare the model for inference and return its path.

    This function handles different scenarios:
    1. An existing GGUF file (via MODEL_FILENAME)
    2. A safetensors model in HuggingFace format (via HF_MODEL_ID or HF_MODEL_URI)
    3. S3 downloads that can be either GGUF (requires MODEL_FILENAME) or safetensors (optional)

    Returns:
        Path to the prepared model file

    Raises:
        RuntimeError: If no model source is provided or no usable model is found
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    gguf_file = os.getenv("MODEL_FILENAME", "").strip() or None
    quantization = os.getenv("QUANTIZATION", "").strip() or None

    model_root = MODELS_DIR / "current"
    if not model_root.exists():
        tmp_root = MODELS_DIR / "download"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

        # Download HuggingFace model
        hf_model_id = os.environ.get("HF_MODEL_ID")
        hf_model_uri = os.environ.get("HF_MODEL_URI")

        if hf_model_id:
            download_hf(repo_id=hf_model_id, dest_dir=tmp_root, filename=gguf_file)
        elif hf_model_uri:
            # For S3 downloads, determine if we need MODEL_FILENAME based on content type
            model_type = _detect_model_type_from_s3_uri(hf_model_uri)

            if model_type == "gguf" and not gguf_file:
                error_msg = "MODEL_FILENAME is required for GGUF downloads from S3. Please specify the GGUF file to download."
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            elif model_type == "unknown" and not gguf_file:
                # For backward compatibility, warn but allow if we can't detect
                print(
                    "WARNING: Could not detect model type from S3 URI. Assuming safetensors format."
                )

            download_s3(s3_uri=hf_model_uri, dest_dir=tmp_root, filename=gguf_file)
        else:
            raise RuntimeError("Either HF_MODEL_ID or HF_MODEL_URI must be provided")

        tmp_root.rename(model_root)

    # Scenario 1: Check for existing GGUF file
    if gguf_file:
        gguf_path = model_root / gguf_file
        if gguf_path.exists() and gguf_path.suffix.lower() == ".gguf":
            return str(gguf_path)
        else:
            raise RuntimeError(
                f"Specified MODEL_FILENAME {gguf_file} not found or not a GGUF file"
            )

    # Scenario 2: Convert HuggingFace safetensors model to GGUF
    if _looks_like_hf_repo(model_root):
        f16_path = _convert_hf_to_gguf(model_root, model_root)
        if quantization:
            q_path = _quantize_gguf(f16_path, quantization)
            return str(q_path)
        return str(f16_path)

    # This should not be reached for S3 GGUF downloads since MODEL_FILENAME is required
    raise RuntimeError(
        "No usable model found. For GGUF files, provide MODEL_FILENAME. For HuggingFace safetensors models, ensure the downloaded files contain config.json or .safetensors files."
    )
