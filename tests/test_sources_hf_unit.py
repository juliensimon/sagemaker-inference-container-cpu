"""
Unit tests for the HuggingFace source module.

This module contains unit tests for all functions in app.sources_hf.py,
including model downloading and authentication handling.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.sources_hf import download_hf


class TestDownloadHf:
    """Test the download_hf function."""

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_full_model_success(self, mock_snapshot_download, temp_dir):
        """Test successful full model download."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        result = download_hf(repo_id=repo_id, dest_dir=dest_dir)

        # Verify snapshot_download was called with correct parameters
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=None,
        )

        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_with_token(self, mock_snapshot_download, temp_dir):
        """Test download with explicit token."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"
        token = "hf_test_token"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        result = download_hf(repo_id=repo_id, dest_dir=dest_dir, token=token)

        # Verify token was passed correctly
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=token,
        )

        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_with_hf_token_env(self, mock_snapshot_download, temp_dir):
        """Test download with HF_TOKEN environment variable."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"
        token = "hf_env_token"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        with patch.dict("os.environ", {"HF_TOKEN": token}):
            result = download_hf(repo_id=repo_id, dest_dir=dest_dir)

        # Verify environment token was used
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=token,
        )

        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_with_huggingface_token_env(
        self, mock_snapshot_download, temp_dir
    ):
        """Test download with HUGGINGFACE_TOKEN environment variable."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"
        token = "huggingface_env_token"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        with patch.dict("os.environ", {"HUGGINGFACE_TOKEN": token}):
            result = download_hf(repo_id=repo_id, dest_dir=dest_dir)

        # Verify environment token was used
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=token,
        )

        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_token_precedence(self, mock_snapshot_download, temp_dir):
        """Test that explicit token takes precedence over environment variables."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"
        explicit_token = "explicit_token"
        env_token = "env_token"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        with patch.dict("os.environ", {"HF_TOKEN": env_token}):
            result = download_hf(
                repo_id=repo_id, dest_dir=dest_dir, token=explicit_token
            )

        # Verify explicit token was used, not environment token
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=explicit_token,
        )

        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_hf_token_precedence_over_huggingface_token(
        self, mock_snapshot_download, temp_dir
    ):
        """Test that HF_TOKEN takes precedence over HUGGINGFACE_TOKEN."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"
        hf_token = "hf_token"
        huggingface_token = "huggingface_token"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        with patch.dict(
            "os.environ", {"HF_TOKEN": hf_token, "HUGGINGFACE_TOKEN": huggingface_token}
        ):
            result = download_hf(repo_id=repo_id, dest_dir=dest_dir)

        # Verify HF_TOKEN was used, not HUGGINGFACE_TOKEN
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=hf_token,
        )

        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_dest_dir_creation(self, mock_snapshot_download, temp_dir):
        """Test that destination directory is created if it doesn't exist."""
        repo_id = "test/model"
        dest_dir = temp_dir / "new" / "download" / "path"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        result = download_hf(repo_id=repo_id, dest_dir=dest_dir)

        # Verify snapshot_download was called with the correct path
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=None,
        )

        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_prints_progress(
        self, mock_snapshot_download, temp_dir, capsys
    ):
        """Test that download progress is printed."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        result = download_hf(repo_id=repo_id, dest_dir=dest_dir)

        captured = capsys.readouterr()
        assert "Downloading model from Hugging Face: test/model" in captured.out
        assert "Successfully downloaded test/model" in captured.out
        # Function returns None
        assert result is None

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_error_prints_message(
        self, mock_snapshot_download, temp_dir, capsys
    ):
        """Test that error messages are printed on download failure."""
        repo_id = "test/model"
        dest_dir = temp_dir / "download"

        # Mock download failure
        mock_snapshot_download.side_effect = Exception("Test error")

        with pytest.raises(
            RuntimeError, match="Failed to download model test/model: Test error"
        ):
            download_hf(repo_id=repo_id, dest_dir=dest_dir)

        captured = capsys.readouterr()
        assert "Failed to download test/model: Test error" in captured.out

    @pytest.mark.unit
    @patch("app.sources_hf.snapshot_download")
    def test_download_hf_with_complex_repo_id(self, mock_snapshot_download, temp_dir):
        """Test download with complex repository ID."""
        repo_id = "arcee-ai/arcee-lite"
        dest_dir = temp_dir / "download"

        # Mock successful download
        mock_snapshot_download.return_value = str(dest_dir)

        result = download_hf(repo_id=repo_id, dest_dir=dest_dir)

        # Verify snapshot_download was called with correct parameters
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            token=None,
        )

        # Function returns None
        assert result is None
