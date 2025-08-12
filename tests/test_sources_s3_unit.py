"""
Unit tests for the S3 source module.

This module contains unit tests for all functions in app.sources_s3.py,
including S3 URI parsing and model downloading.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.sources_s3 import _parse_s3_uri, download_s3


class TestParseS3Uri:
    """Test the _parse_s3_uri function."""

    @pytest.mark.unit
    def test_parse_s3_uri_basic(self):
        """Test parsing basic S3 URI."""
        s3_uri = "s3://my-bucket/path/to/files"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my-bucket"
        assert prefix == "path/to/files"

    @pytest.mark.unit
    def test_parse_s3_uri_root_bucket(self):
        """Test parsing S3 URI with root bucket."""
        s3_uri = "s3://my-bucket/"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my-bucket"
        assert prefix == ""

    @pytest.mark.unit
    def test_parse_s3_uri_no_prefix(self):
        """Test parsing S3 URI without prefix."""
        s3_uri = "s3://my-bucket"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my-bucket"
        assert prefix == ""

    @pytest.mark.unit
    def test_parse_s3_uri_single_file(self):
        """Test parsing S3 URI pointing to single file."""
        s3_uri = "s3://my-bucket/model.gguf"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my-bucket"
        assert prefix == "model.gguf"

    @pytest.mark.unit
    def test_parse_s3_uri_nested_path(self):
        """Test parsing S3 URI with deeply nested path."""
        s3_uri = "s3://my-bucket/models/llama/7b/gguf/model.gguf"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my-bucket"
        assert prefix == "models/llama/7b/gguf/model.gguf"

    @pytest.mark.unit
    def test_parse_s3_uri_with_special_characters(self):
        """Test parsing S3 URI with special characters in path."""
        s3_uri = "s3://my-bucket/path/with spaces and-special_chars/file.txt"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my-bucket"
        assert prefix == "path/with spaces and-special_chars/file.txt"

    @pytest.mark.unit
    def test_parse_s3_uri_invalid_format(self):
        """Test parsing of invalid S3 URI formats."""
        invalid_uris = [
            "not-s3://bucket/key",
            "http://bucket/key",
            "ftp://bucket/key",
            "s3:///key",  # Missing bucket
        ]

        for uri in invalid_uris:
            with pytest.raises(ValueError, match="Invalid S3 URI"):
                _parse_s3_uri(uri)

    @pytest.mark.unit
    def test_parse_s3_uri_none(self):
        """Test parsing of None S3 URI."""
        with pytest.raises(TypeError, match="expected string or bytes-like object"):
            _parse_s3_uri(None)


class TestDownloadS3:
    """Test the download_s3 function."""

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_single_file(self, mock_boto3_client, temp_dir):
        """Test downloading a single file from S3."""
        s3_uri = "s3://my-bucket/model.gguf"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": [{"Key": "model.gguf"}]}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        # Verify S3 client was called correctly
        mock_boto3_client.assert_called_once_with("s3")
        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
        mock_paginator.paginate.assert_called_once_with(
            Bucket="my-bucket", Prefix="model.gguf"
        )
        mock_s3.download_file.assert_called_once_with(
            "my-bucket", "model.gguf", str(dest_dir / "model.gguf")
        )

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_directory(self, mock_boto3_client, temp_dir):
        """Test downloading multiple files from S3 directory."""
        s3_uri = "s3://my-bucket/models/"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "models/config.json"},
                {"Key": "models/model.safetensors"},
                {"Key": "models/tokenizer.json"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        # Verify all files were downloaded
        assert mock_s3.download_file.call_count == 3
        expected_calls = [
            (("my-bucket", "models/config.json", str(dest_dir / "config.json")),),
            (
                (
                    "my-bucket",
                    "models/model.safetensors",
                    str(dest_dir / "model.safetensors"),
                ),
            ),
            (("my-bucket", "models/tokenizer.json", str(dest_dir / "tokenizer.json")),),
        ]
        mock_s3.download_file.assert_has_calls(expected_calls, any_order=True)

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_nested_directory(self, mock_boto3_client, temp_dir):
        """Test downloading from nested S3 directory structure."""
        s3_uri = "s3://my-bucket/models/llama/7b/"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "models/llama/7b/config.json"},
                {"Key": "models/llama/7b/model-00001-of-00002.safetensors"},
                {"Key": "models/llama/7b/model-00002-of-00002.safetensors"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        # Verify files were downloaded with correct relative paths
        assert mock_s3.download_file.call_count == 3
        expected_calls = [
            (
                (
                    "my-bucket",
                    "models/llama/7b/config.json",
                    str(dest_dir / "config.json"),
                ),
            ),
            (
                (
                    "my-bucket",
                    "models/llama/7b/model-00001-of-00002.safetensors",
                    str(dest_dir / "model-00001-of-00002.safetensors"),
                ),
            ),
            (
                (
                    "my-bucket",
                    "models/llama/7b/model-00002-of-00002.safetensors",
                    str(dest_dir / "model-00002-of-00002.safetensors"),
                ),
            ),
        ]
        mock_s3.download_file.assert_has_calls(expected_calls, any_order=True)

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_no_files_found(self, mock_boto3_client, temp_dir):
        """Test error handling when no files are found in S3."""
        s3_uri = "s3://my-bucket/empty-directory/"
        dest_dir = temp_dir / "download"

        # Mock S3 client with no contents
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": []}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        with pytest.raises(ValueError, match="No files found in S3 URI"):
            download_s3(s3_uri, dest_dir)

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_only_directories(self, mock_boto3_client, temp_dir):
        """Test handling when S3 contains only directories (no files)."""
        s3_uri = "s3://my-bucket/directories-only/"
        dest_dir = temp_dir / "download"

        # Mock S3 client with only directory objects
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "directories-only/subdir1/"},
                {"Key": "directories-only/subdir2/"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        with pytest.raises(ValueError, match="No files found in S3 URI"):
            download_s3(s3_uri, dest_dir)

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_download_failure(self, mock_boto3_client, temp_dir):
        """Test error handling when S3 download fails."""
        s3_uri = "s3://my-bucket/model.gguf"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": [{"Key": "model.gguf"}]}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.side_effect = Exception("S3 download failed")
        mock_boto3_client.return_value = mock_s3

        with pytest.raises(Exception, match="S3 download failed"):
            download_s3(s3_uri, dest_dir)

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_list_objects_failure(self, mock_boto3_client, temp_dir):
        """Test error handling when listing S3 objects fails."""
        s3_uri = "s3://my-bucket/model.gguf"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate.side_effect = Exception("List objects failed")
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        with pytest.raises(Exception, match="List objects failed"):
            download_s3(s3_uri, dest_dir)

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_creates_dest_dir(self, mock_boto3_client, temp_dir):
        """Test that destination directory is created if it doesn't exist."""
        s3_uri = "s3://my-bucket/model.gguf"
        dest_dir = temp_dir / "new" / "download" / "path"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": [{"Key": "model.gguf"}]}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        # Verify the directory was created
        assert dest_dir.exists()
        assert dest_dir.is_dir()

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_prints_progress(self, mock_boto3_client, temp_dir, capsys):
        """Test that download progress is printed."""
        s3_uri = "s3://my-bucket/models/"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "models/config.json"},
                {"Key": "models/model.safetensors"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        captured = capsys.readouterr()
        assert "Downloading from S3: bucket=my-bucket, prefix=models/" in captured.out
        assert "Found 2 object(s) to download" in captured.out
        assert "Downloading 2 files to directory structure" in captured.out
        assert "Successfully downloaded all files" in captured.out

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_single_file_prints_progress(
        self, mock_boto3_client, temp_dir, capsys
    ):
        """Test that single file download progress is printed."""
        s3_uri = "s3://my-bucket/model.gguf"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": [{"Key": "model.gguf"}]}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        captured = capsys.readouterr()
        assert (
            "Downloading from S3: bucket=my-bucket, prefix=model.gguf" in captured.out
        )
        assert "Found 1 object(s) to download" in captured.out
        assert "Downloading single file to:" in captured.out
        assert "Successfully downloaded:" in captured.out

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_with_special_characters_in_keys(
        self, mock_boto3_client, temp_dir
    ):
        """Test downloading files with special characters in S3 keys."""
        s3_uri = "s3://my-bucket/models/"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "models/file with spaces.txt"},
                {"Key": "models/file-with-dashes.json"},
                {"Key": "models/file_with_underscores.gguf"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        # Verify files were downloaded with correct names
        expected_calls = [
            (
                (
                    "my-bucket",
                    "models/file with spaces.txt",
                    str(dest_dir / "file with spaces.txt"),
                ),
            ),
            (
                (
                    "my-bucket",
                    "models/file-with-dashes.json",
                    str(dest_dir / "file-with-dashes.json"),
                ),
            ),
            (
                (
                    "my-bucket",
                    "models/file_with_underscores.gguf",
                    str(dest_dir / "file_with_underscores.gguf"),
                ),
            ),
        ]
        mock_s3.download_file.assert_has_calls(expected_calls, any_order=True)

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_handles_empty_prefix(self, mock_boto3_client, temp_dir):
        """Test downloading from S3 bucket root."""
        s3_uri = "s3://my-bucket"
        dest_dir = temp_dir / "download"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": [{"Key": "model.gguf"}]}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        # Verify the download was called correctly
        mock_s3.download_file.assert_called_once_with(
            "my-bucket", "model.gguf", str(dest_dir / "model.gguf")
        )

    @pytest.mark.unit
    @patch("boto3.client")
    def test_download_s3_handles_large_number_of_files(
        self, mock_boto3_client, temp_dir
    ):
        """Test downloading a large number of files from S3."""
        s3_uri = "s3://my-bucket/models/"
        dest_dir = temp_dir / "download"

        # Mock S3 client with many files
        mock_s3 = Mock()
        mock_paginator = Mock()

        # Create many mock files
        files = [{"Key": f"models/file_{i:04d}.txt"} for i in range(100)]
        mock_page = {"Contents": files}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.return_value = None
        mock_boto3_client.return_value = mock_s3

        download_s3(s3_uri, dest_dir)

        # Verify all files were downloaded
        assert mock_s3.download_file.call_count == 100

        # Verify a few specific calls
        mock_s3.download_file.assert_any_call(
            "my-bucket", "models/file_0000.txt", str(dest_dir / "file_0000.txt")
        )
        mock_s3.download_file.assert_any_call(
            "my-bucket", "models/file_0099.txt", str(dest_dir / "file_0099.txt")
        )


class TestS3UriEdgeCases:
    """Test edge cases for S3 URI handling."""

    @pytest.mark.unit
    def test_parse_s3_uri_with_numbers(self):
        """Test parsing S3 URI with numbers in bucket and path."""
        s3_uri = "s3://bucket-123/path/456/file.txt"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "bucket-123"
        assert prefix == "path/456/file.txt"

    @pytest.mark.unit
    def test_parse_s3_uri_with_underscores(self):
        """Test parsing S3 URI with underscores in bucket and path."""
        s3_uri = "s3://my_bucket/path_with_underscores/file.txt"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my_bucket"
        assert prefix == "path_with_underscores/file.txt"

    @pytest.mark.unit
    def test_parse_s3_uri_with_dots(self):
        """Test parsing S3 URI with dots in bucket and path."""
        s3_uri = "s3://my.bucket.com/path/to/file.txt"
        bucket, prefix = _parse_s3_uri(s3_uri)
        assert bucket == "my.bucket.com"
        assert prefix == "path/to/file.txt"
