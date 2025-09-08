#!/usr/bin/env python3
"""
Hugging Face model download utilities.

This module provides functions for downloading models from Hugging Face Hub,
including support for gated/private models and specific file downloads.
It handles authentication, download resumption, and provides detailed
progress information.

Environment Variables:
    HF_TOKEN: Hugging Face authentication token
    HUGGINGFACE_TOKEN: Alternative Hugging Face token environment variable
"""

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def download_hf(
    repo_id: str,
    dest_dir: Path,
    token: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Download a model from Hugging Face Hub

    Args:
        repo_id: Hugging Face repository ID (e.g., "arcee-ai/AFM-4.5B")
        dest_dir: Destination directory for the downloaded model
        token: Hugging Face token for gated/private models
        filename: Specific file to download (e.g., "model.gguf")

    Raises:
        RuntimeError: If download fails
    """
    print(f"Downloading model from Hugging Face: {repo_id}")
    if filename:
        print(f"Downloading specific file: {filename}")
    print(f"Destination: {dest_dir}")

    # Get token from environment if not provided
    if not token:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    try:
        if filename:
            # Download only the specific file
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest_dir),
                token=token,
            )
            print(
                f"✅ Successfully downloaded {filename} from {repo_id} to {file_path}"
            )
        else:
            # Download the entire model
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest_dir),
                token=token,
            )
            print(f"✅ Successfully downloaded {repo_id} to {dest_dir}")

    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        raise RuntimeError(f"Failed to download model {repo_id}: {e}")
