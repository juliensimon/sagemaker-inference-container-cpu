"""
S3 model download utilities.

This module provides functions for downloading models from Amazon S3,
supporting both single file downloads and directory structures.
It handles S3 URI parsing, pagination for large directories, and
provides detailed progress information.

The module supports downloading from S3 URIs in the format:
s3://bucket-name/path/to/model/files/
"""

import os
import re
from pathlib import Path
from typing import Tuple

import boto3


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI into bucket and prefix components.

    Args:
        s3_uri: S3 URI in format s3://bucket-name/path/to/files

    Returns:
        Tuple of (bucket_name, prefix)

    Raises:
        ValueError: If the S3 URI format is invalid
    """
    m = re.match(r"^s3://([^/]+)/?(.*)$", s3_uri)
    if not m:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket, prefix = m.group(1), m.group(2)
    return bucket, prefix


def download_s3(s3_uri: str, dest_dir: Path) -> None:
    """
    Download files from an S3 URI to a local directory.

    This function handles both single file downloads and directory structures.
    For single files, it downloads directly to the destination directory.
    For directories, it preserves the directory structure.

    Args:
        s3_uri: S3 URI pointing to the model files (e.g., s3://bucket/model/)
        dest_dir: Local directory to download files to

    Raises:
        ValueError: If S3 URI is invalid or no files are found
        Exception: If download fails due to S3 permissions or network issues
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    bucket, prefix = _parse_s3_uri(s3_uri)

    print(f"Downloading from S3: bucket={bucket}, prefix={prefix}")

    # Check if this is a single file or directory
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            objects.append(key)

    if not objects:
        raise ValueError(f"No files found in S3 URI: {s3_uri}")

    print(f"Found {len(objects)} object(s) to download")

    # If it's a single file, download directly to dest_dir
    if len(objects) == 1 and objects[0] == prefix:
        # Single file download
        filename = os.path.basename(prefix) if prefix else "model.gguf"
        target = dest_dir / filename
        print(f"Downloading single file to: {target}")
        s3.download_file(bucket, prefix, str(target))
        print(f"Successfully downloaded: {target}")
    else:
        # Multiple files or directory structure
        print(f"Downloading {len(objects)} files to directory structure")
        for key in objects:
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/") if prefix else key
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {key} to {target}")
            s3.download_file(bucket, key, str(target))
        print("Successfully downloaded all files")
