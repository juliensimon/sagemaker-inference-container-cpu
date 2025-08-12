"""
Unit tests for the model manager module.

This module contains unit tests for all functions in app.model_manager.py,
including model preparation, conversion, quantization, and utility functions.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.model_manager import (
    LLAMACPP_DIR,
    MODELS_DIR,
    _convert_hf_to_gguf,
    _detect_model_type_from_s3_uri,
    _find_convert_script,
    _find_quantize_binary,
    _looks_like_hf_repo,
    _quantize_gguf,
    prepare_model_and_get_path,
)


class TestFindQuantizeBinary:
    """Test the _find_quantize_binary function."""

    @pytest.mark.unit
    @patch("shutil.which")
    def test_find_quantize_binary_found(self, mock_which):
        """Test finding quantize binary when it exists."""
        mock_which.return_value = "/usr/bin/llama-quantize"
        result = _find_quantize_binary()
        assert result == "/usr/bin/llama-quantize"
        mock_which.assert_called_once_with("llama-quantize")

    @pytest.mark.unit
    @patch("shutil.which")
    def test_find_quantize_binary_not_found(self, mock_which):
        """Test finding quantize binary when it doesn't exist."""
        mock_which.return_value = None
        result = _find_quantize_binary()
        assert result is None


class TestFindConvertScript:
    """Test the _find_convert_script function."""

    @pytest.mark.unit
    @patch("pathlib.Path.exists")
    def test_find_convert_script_found(self, mock_exists):
        """Test finding convert script when it exists."""
        mock_exists.return_value = True
        result = _find_convert_script()
        expected_path = LLAMACPP_DIR / "convert_hf_to_gguf.py"
        assert result == expected_path

    @pytest.mark.unit
    @patch("pathlib.Path.exists")
    def test_find_convert_script_not_found(self, mock_exists):
        """Test finding convert script when it doesn't exist."""
        mock_exists.return_value = False
        result = _find_convert_script()
        assert result is None


class TestConvertHfToGguf:
    """Test the _convert_hf_to_gguf function."""

    @pytest.mark.unit
    @patch("app.model_manager._find_convert_script")
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_convert_hf_to_gguf_success(
        self, mock_mkdir, mock_run, mock_find_script, temp_dir
    ):
        """Test successful HuggingFace to GGUF conversion."""
        source_dir = temp_dir / "source"
        out_dir = temp_dir / "output"
        script_path = temp_dir / "convert_hf_to_gguf.py"

        mock_find_script.return_value = script_path
        mock_run.return_value = Mock(returncode=0)

        result = _convert_hf_to_gguf(source_dir, out_dir)

        expected_outfile = out_dir / "model-f16.gguf"
        assert result == expected_outfile

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_run.assert_called_once()

        # Check the command arguments
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/opt/venv/bin/python3"
        assert call_args[1] == str(script_path)
        assert "--outtype" in call_args
        assert "f16" in call_args
        assert "--outfile" in call_args
        assert str(expected_outfile) in call_args
        assert str(source_dir) in call_args

    @pytest.mark.unit
    @patch("app.model_manager._find_convert_script")
    def test_convert_hf_to_gguf_script_not_found(self, mock_find_script):
        """Test error when convert script is not found."""
        mock_find_script.return_value = None

        with pytest.raises(RuntimeError, match="convert_hf_to_gguf.py not found"):
            _convert_hf_to_gguf(Path("/source"), Path("/output"))

    @pytest.mark.unit
    @patch("app.model_manager._find_convert_script")
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_convert_hf_to_gguf_subprocess_failure(
        self, mock_mkdir, mock_run, mock_find_script, temp_dir
    ):
        """Test error when subprocess conversion fails."""
        source_dir = temp_dir / "source"
        out_dir = temp_dir / "output"
        script_path = temp_dir / "convert_hf_to_gguf.py"

        mock_find_script.return_value = script_path
        mock_run.side_effect = subprocess.CalledProcessError(1, "convert_script")

        with pytest.raises(subprocess.CalledProcessError):
            _convert_hf_to_gguf(source_dir, out_dir)


class TestQuantizeGguf:
    """Test the _quantize_gguf function."""

    @pytest.mark.unit
    @patch("app.model_manager._find_quantize_binary")
    @patch("subprocess.run")
    def test_quantize_gguf_success(self, mock_run, mock_find_binary, temp_dir):
        """Test successful GGUF quantization."""
        src_path = temp_dir / "model.gguf"
        qtype = "q4_k_m"

        mock_find_binary.return_value = "/usr/bin/llama-quantize"
        mock_run.return_value = Mock(returncode=0)

        result = _quantize_gguf(src_path, qtype)

        expected_out_path = temp_dir / "model.q4_k_m.gguf"
        assert result == expected_out_path

        mock_run.assert_called_once()

        # Check the command arguments
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/usr/bin/llama-quantize"
        assert call_args[1] == str(src_path)
        assert call_args[2] == str(expected_out_path)
        assert call_args[3] == qtype

    @pytest.mark.unit
    @patch("app.model_manager._find_quantize_binary")
    def test_quantize_gguf_binary_not_found(self, mock_find_binary):
        """Test error when quantize binary is not found."""
        mock_find_binary.return_value = None

        with pytest.raises(RuntimeError, match="llama-quantize binary not found"):
            _quantize_gguf(Path("/model.gguf"), "q4_k_m")

    @pytest.mark.unit
    @patch("app.model_manager._find_quantize_binary")
    @patch("subprocess.run")
    def test_quantize_gguf_subprocess_failure(
        self, mock_run, mock_find_binary, temp_dir
    ):
        """Test error when subprocess quantization fails."""
        src_path = temp_dir / "model.gguf"
        qtype = "q4_k_m"

        mock_find_binary.return_value = "/usr/bin/llama-quantize"
        mock_run.side_effect = subprocess.CalledProcessError(1, "quantize")

        with pytest.raises(subprocess.CalledProcessError):
            _quantize_gguf(src_path, qtype)

    @pytest.mark.unit
    @patch("app.model_manager._find_quantize_binary")
    @patch("subprocess.run")
    def test_quantize_gguf_filename_handling(
        self, mock_run, mock_find_binary, temp_dir
    ):
        """Test quantization with different filename patterns."""
        src_path = temp_dir / "complex-model-name.f16.gguf"
        qtype = "q8_0"

        mock_find_binary.return_value = "/usr/bin/llama-quantize"
        mock_run.return_value = Mock(returncode=0)

        result = _quantize_gguf(src_path, qtype)

        expected_out_path = temp_dir / "complex-model-name.f16.q8_0.gguf"
        assert result == expected_out_path


class TestLooksLikeHfRepo:
    """Test the _looks_like_hf_repo function."""

    @pytest.mark.unit
    def test_looks_like_hf_repo_with_config_json(self, temp_dir):
        """Test that a directory with config.json is recognized as HF repo."""
        model_dir = temp_dir / "test-model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()

        # Mock the exists method on Path instances
        original_exists = Path.exists

        def mock_exists(self):
            return str(self).endswith("config.json")

        Path.exists = mock_exists

        try:
            result = _looks_like_hf_repo(model_dir)
            assert result is True
        finally:
            Path.exists = original_exists

    @pytest.mark.unit
    def test_looks_like_hf_repo_with_safetensors(self, temp_dir):
        """Test that a directory with safetensors files is recognized as HF repo."""
        model_dir = temp_dir / "test-model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()

        # Mock the exists method on Path instances
        original_exists = Path.exists

        def mock_exists(self):
            return str(self).endswith("config.json")

        Path.exists = mock_exists

        try:
            with patch("pathlib.Path.glob") as mock_glob:
                mock_glob.return_value = [model_dir / "model.safetensors"]
                result = _looks_like_hf_repo(model_dir)
                assert result is True
        finally:
            Path.exists = original_exists

    @pytest.mark.unit
    def test_looks_like_hf_repo_invalid(self, temp_dir):
        """Test that a directory without HF files is not recognized as HF repo."""
        model_dir = temp_dir / "test-model"
        model_dir.mkdir()
        (model_dir / "random.txt").touch()

        # Mock the exists method on Path instances
        original_exists = Path.exists

        def mock_exists(self):
            return False

        Path.exists = mock_exists

        try:
            with patch("pathlib.Path.glob") as mock_glob:
                mock_glob.return_value = []
                result = _looks_like_hf_repo(model_dir)
                assert result is False
        finally:
            Path.exists = original_exists


class TestPrepareModelAndGetPath:
    """Test the prepare_model_and_get_path function."""

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_hf")
    def test_prepare_model_gguf_filename(
        self, mock_download, mock_rename, mock_exists, mock_mkdir, temp_dir
    ):
        """Test model preparation with existing GGUF filename."""
        # Mock environment
        with patch.dict(
            "os.environ",
            {"MODEL_FILENAME": "test_model.gguf", "HF_MODEL_ID": "test/model"},
        ):
            # Mock file existence - assume files exist for this test
            mock_exists.return_value = True

            with patch("app.model_manager.MODELS_DIR", temp_dir):
                result = prepare_model_and_get_path()
                expected_path = str(temp_dir / "current" / "test_model.gguf")
                assert result == expected_path

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_hf")
    def test_prepare_model_gguf_filename_not_found(
        self, mock_download, mock_rename, mock_exists, mock_mkdir
    ):
        """Test error when specified GGUF filename is not found."""
        with patch.dict(
            "os.environ",
            {"MODEL_FILENAME": "missing_model.gguf", "HF_MODEL_ID": "test/model"},
        ):
            mock_exists.return_value = False

            with patch("app.model_manager.MODELS_DIR", Path("/tmp")):
                with pytest.raises(
                    RuntimeError,
                    match="Specified MODEL_FILENAME missing_model.gguf not found",
                ):
                    prepare_model_and_get_path()

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_hf")
    @patch("app.model_manager._looks_like_hf_repo")
    @patch("app.model_manager._convert_hf_to_gguf")
    @patch("app.model_manager._quantize_gguf")
    def test_prepare_model_hf_conversion(
        self,
        mock_quantize,
        mock_convert,
        mock_looks_like,
        mock_download,
        mock_rename,
        mock_exists,
        mock_mkdir,
        temp_dir,
        mock_models_dir,
    ):
        """Test model preparation with HF conversion."""

        # Mock file existence - no current model exists
        def mock_exists_func(self):
            return False

        # Mock the exists method on Path instances
        original_exists = Path.exists
        Path.exists = mock_exists_func

        try:
            # Mock HF repo detection
            mock_looks_like.return_value = True

            # Mock conversion
            converted_file = temp_dir / "converted.gguf"
            mock_convert.return_value = converted_file

            with patch.dict("os.environ", {"HF_MODEL_ID": "test/model"}):
                result = prepare_model_and_get_path()

                # Verify conversion was called
                mock_convert.assert_called_once()

                # Verify result
                assert result == str(converted_file)
        finally:
            Path.exists = original_exists

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_hf")
    @patch("app.model_manager._looks_like_hf_repo")
    @patch("app.model_manager._convert_hf_to_gguf")
    @patch("app.model_manager._quantize_gguf")
    def test_prepare_model_hf_conversion_with_quantization(
        self,
        mock_quantize,
        mock_convert,
        mock_looks_like,
        mock_download,
        mock_rename,
        mock_exists,
        mock_mkdir,
        temp_dir,
        mock_models_dir,
    ):
        """Test model preparation with HF conversion and quantization."""

        # Mock file existence - no current model exists
        def mock_exists_func(self):
            return False

        # Mock the exists method on Path instances
        original_exists = Path.exists
        Path.exists = mock_exists_func

        try:
            # Mock HF repo detection
            mock_looks_like.return_value = True

            # Mock conversion and quantization
            converted_file = temp_dir / "converted.gguf"
            quantized_file = temp_dir / "converted.Q4_K_M.gguf"
            mock_convert.return_value = converted_file
            mock_quantize.return_value = quantized_file

            with patch.dict(
                "os.environ", {"HF_MODEL_ID": "test/model", "QUANTIZATION": "Q4_K_M"}
            ):
                result = prepare_model_and_get_path()

                # Verify conversion was called
                mock_convert.assert_called_once()

                # Verify quantization was called
                mock_quantize.assert_called_once_with(converted_file, "Q4_K_M")

                # Verify result
                assert result == str(quantized_file)
        finally:
            Path.exists = original_exists

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    def test_prepare_model_no_source_provided(
        self, mock_rename, mock_exists, mock_mkdir
    ):
        """Test error when no model source is provided."""
        with patch.dict("os.environ", {}, clear=True):
            mock_exists.return_value = False

            with patch("app.model_manager.MODELS_DIR", Path("/tmp")):
                with pytest.raises(
                    RuntimeError,
                    match="Either HF_MODEL_ID or HF_MODEL_URI must be provided",
                ):
                    prepare_model_and_get_path()

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_hf")
    @patch("app.model_manager._looks_like_hf_repo")
    def test_prepare_model_no_usable_model(
        self, mock_looks_like, mock_download, mock_rename, mock_exists, mock_mkdir
    ):
        """Test error when no usable model is found."""
        with patch.dict("os.environ", {"HF_MODEL_ID": "test/model"}):
            mock_exists.return_value = False
            mock_looks_like.return_value = False

            with patch("app.model_manager.MODELS_DIR", Path("/tmp")):
                with pytest.raises(RuntimeError, match="No usable model found"):
                    prepare_model_and_get_path()

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_s3")
    def test_prepare_model_s3_source_with_model_filename(
        self, mock_download_s3, mock_rename, mock_exists, mock_mkdir, temp_dir
    ):
        """Test model preparation with S3 source and MODEL_FILENAME."""
        with patch.dict(
            "os.environ",
            {"HF_MODEL_URI": "s3://bucket/model/", "MODEL_FILENAME": "test.gguf"},
        ):
            # Mock file existence - model doesn't exist initially, then GGUF file exists after download
            call_count = 0

            def mock_exists_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # First call: model_root.exists() returns False (no current model)
                if call_count == 1:
                    return False
                # Second call: gguf_path.exists() returns True (GGUF file exists)
                elif call_count == 2:
                    return True
                # Any additional calls return True
                return True

            mock_exists.side_effect = mock_exists_side_effect

            with patch("app.model_manager.MODELS_DIR", temp_dir):
                result = prepare_model_and_get_path()
                expected_path = str(temp_dir / "current" / "test.gguf")
                assert result == expected_path

                mock_download_s3.assert_called_once()

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_s3")
    @patch("app.model_manager._detect_model_type_from_s3_uri")
    def test_prepare_model_s3_source_missing_model_filename(
        self,
        mock_detect_type,
        mock_download_s3,
        mock_rename,
        mock_exists,
        mock_mkdir,
        temp_dir,
    ):
        """Test error when MODEL_FILENAME is missing for GGUF S3 source."""
        with patch.dict("os.environ", {"HF_MODEL_URI": "s3://bucket/model/"}):
            # Mock detection to return gguf so it requires MODEL_FILENAME
            mock_detect_type.return_value = "gguf"
            mock_exists.return_value = False

            with patch("app.model_manager.MODELS_DIR", temp_dir):
                with pytest.raises(
                    RuntimeError,
                    match="MODEL_FILENAME is required for GGUF downloads from S3",
                ):
                    prepare_model_and_get_path()

                # Download should not be called since we error before that
                mock_download_s3.assert_not_called()

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_s3")
    @patch("app.model_manager._detect_model_type_from_s3_uri")
    @patch("app.model_manager._looks_like_hf_repo")
    @patch("app.model_manager._convert_hf_to_gguf")
    def test_prepare_model_s3_safetensors_without_model_filename(
        self,
        mock_convert,
        mock_looks_like,
        mock_detect_type,
        mock_download_s3,
        mock_rename,
        mock_exists,
        mock_mkdir,
        temp_dir,
    ):
        """Test safetensors model from S3 without MODEL_FILENAME."""
        with patch.dict(
            "os.environ", {"HF_MODEL_URI": "s3://bucket/safetensors-model/"}
        ):
            # Mock detection to return safetensors
            mock_detect_type.return_value = "safetensors"

            # Mock file existence - model doesn't exist initially, then we have HF repo
            call_count = 0

            def mock_exists_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # First call: model_root.exists() returns False (no current model)
                return False

            mock_exists.side_effect = mock_exists_side_effect

            # Mock HF repo detection
            mock_looks_like.return_value = True

            # Mock conversion
            converted_file = temp_dir / "converted.gguf"
            mock_convert.return_value = converted_file

            with patch("app.model_manager.MODELS_DIR", temp_dir):
                result = prepare_model_and_get_path()

                # Should succeed without MODEL_FILENAME
                mock_download_s3.assert_called_once()
                mock_detect_type.assert_called_once_with(
                    "s3://bucket/safetensors-model/"
                )
                mock_convert.assert_called_once()
                assert result == str(converted_file)

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_s3")
    @patch("app.model_manager._detect_model_type_from_s3_uri")
    def test_prepare_model_s3_gguf_without_model_filename_fails(
        self,
        mock_detect_type,
        mock_download_s3,
        mock_rename,
        mock_exists,
        mock_mkdir,
        temp_dir,
    ):
        """Test GGUF model from S3 without MODEL_FILENAME fails."""
        with patch.dict("os.environ", {"HF_MODEL_URI": "s3://bucket/gguf-model/"}):
            # Mock detection to return gguf
            mock_detect_type.return_value = "gguf"

            mock_exists.return_value = False

            with patch("app.model_manager.MODELS_DIR", temp_dir):
                with pytest.raises(
                    RuntimeError,
                    match="MODEL_FILENAME is required for GGUF downloads from S3",
                ):
                    prepare_model_and_get_path()

                # Download should not be called since we error before that
                mock_download_s3.assert_not_called()

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_s3")
    @patch("app.model_manager._detect_model_type_from_s3_uri")
    @patch("app.model_manager._looks_like_hf_repo")
    @patch("app.model_manager._convert_hf_to_gguf")
    def test_prepare_model_s3_unknown_type_without_model_filename_warning(
        self,
        mock_convert,
        mock_looks_like,
        mock_detect_type,
        mock_download_s3,
        mock_rename,
        mock_exists,
        mock_mkdir,
        temp_dir,
        capsys,
    ):
        """Test unknown model type from S3 without MODEL_FILENAME shows warning."""
        with patch.dict("os.environ", {"HF_MODEL_URI": "s3://bucket/unknown-model/"}):
            # Mock detection to return unknown
            mock_detect_type.return_value = "unknown"

            # Mock file existence - model doesn't exist initially, then we have HF repo
            mock_exists.return_value = False

            # Mock HF repo detection
            mock_looks_like.return_value = True

            # Mock conversion
            converted_file = temp_dir / "converted.gguf"
            mock_convert.return_value = converted_file

            with patch("app.model_manager.MODELS_DIR", temp_dir):
                result = prepare_model_and_get_path()

                # Should succeed with warning
                captured = capsys.readouterr()
                assert (
                    "WARNING: Could not detect model type from S3 URI" in captured.out
                )
                assert "Assuming safetensors format" in captured.out

                mock_download_s3.assert_called_once()
                mock_convert.assert_called_once()
                assert result == str(converted_file)

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_hf")
    @patch("app.model_manager._looks_like_hf_repo")
    @patch("app.model_manager._convert_hf_to_gguf")
    @patch("app.model_manager._quantize_gguf")
    def test_quantization_workflow_after_conversion(
        self,
        mock_quantize,
        mock_convert,
        mock_looks_like,
        mock_download,
        mock_rename,
        mock_exists,
        mock_mkdir,
        temp_dir,
        mock_models_dir,
    ):
        """Test that quantization happens after conversion with proper workflow."""

        # Mock file existence - no current model exists
        def mock_exists_func(self):
            return False

        # Mock the exists method on Path instances
        original_exists = Path.exists
        Path.exists = mock_exists_func

        try:
            # Mock HF repo detection
            mock_looks_like.return_value = True

            # Mock conversion and quantization
            converted_file = temp_dir / "converted.gguf"
            quantized_file = temp_dir / "converted.Q4_K_M.gguf"
            mock_convert.return_value = converted_file
            mock_quantize.return_value = quantized_file

            with patch.dict(
                "os.environ", {"HF_MODEL_ID": "test/model", "QUANTIZATION": "Q4_K_M"}
            ):
                result = prepare_model_and_get_path()

                # Verify conversion was called first
                mock_convert.assert_called_once()

                # Verify quantization was called with converted file
                mock_quantize.assert_called_once_with(converted_file, "Q4_K_M")

                # Verify result is the quantized file
                assert result == str(quantized_file)
        finally:
            Path.exists = original_exists

    @pytest.mark.unit
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.rename")
    @patch("app.model_manager.download_hf")
    @patch("app.model_manager._looks_like_hf_repo")
    def test_no_usable_model_found_after_download(
        self,
        mock_looks_like,
        mock_download,
        mock_rename,
        mock_exists,
        mock_mkdir,
        mock_models_dir,
    ):
        """Test error when no usable model is found after download."""

        # Mock file existence - no current model exists
        def mock_exists_func(self):
            return False

        # Mock the exists method on Path instances
        original_exists = Path.exists
        Path.exists = mock_exists_func

        try:
            # Mock HF repo detection - not a valid HF repo
            mock_looks_like.return_value = False

            with patch.dict("os.environ", {"HF_MODEL_ID": "test/model"}):
                with pytest.raises(RuntimeError, match="No usable model found"):
                    prepare_model_and_get_path()
        finally:
            Path.exists = original_exists


class TestDetectModelTypeFromS3Uri:
    """Test the _detect_model_type_from_s3_uri function."""

    @pytest.mark.unit
    @patch("boto3.client")
    def test_detect_gguf_files(self, mock_boto3_client):
        """Test detection of GGUF files in S3."""
        s3_uri = "s3://bucket/gguf-models/"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "gguf-models/model.gguf"},
                {"Key": "gguf-models/other.txt"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        result = _detect_model_type_from_s3_uri(s3_uri)
        assert result == "gguf"

    @pytest.mark.unit
    @patch("boto3.client")
    def test_detect_safetensors_files(self, mock_boto3_client):
        """Test detection of safetensors files in S3."""
        s3_uri = "s3://bucket/safetensors-models/"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "safetensors-models/config.json"},
                {"Key": "safetensors-models/model.safetensors"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        result = _detect_model_type_from_s3_uri(s3_uri)
        assert result == "safetensors"

    @pytest.mark.unit
    @patch("boto3.client")
    def test_detect_unknown_files(self, mock_boto3_client):
        """Test detection when no recognizable files are found."""
        s3_uri = "s3://bucket/unknown-models/"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "unknown-models/README.md"},
                {"Key": "unknown-models/other.txt"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        result = _detect_model_type_from_s3_uri(s3_uri)
        assert result == "unknown"

    @pytest.mark.unit
    @patch("boto3.client")
    def test_detect_prefers_gguf_over_safetensors(self, mock_boto3_client):
        """Test that GGUF is detected first when both types are present."""
        s3_uri = "s3://bucket/mixed-models/"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {
            "Contents": [
                {"Key": "mixed-models/model.gguf"},
                {"Key": "mixed-models/model.safetensors"},
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        result = _detect_model_type_from_s3_uri(s3_uri)
        assert result == "gguf"

    @pytest.mark.unit
    @patch("boto3.client")
    def test_detect_handles_s3_error(self, mock_boto3_client):
        """Test handling of S3 errors during detection."""
        s3_uri = "s3://bucket/models/"

        # Mock S3 client to raise exception
        mock_boto3_client.side_effect = Exception("S3 error")

        result = _detect_model_type_from_s3_uri(s3_uri)
        assert result == "unknown"

    @pytest.mark.unit
    @patch("boto3.client")
    def test_detect_empty_s3_location(self, mock_boto3_client):
        """Test detection when S3 location is empty."""
        s3_uri = "s3://bucket/empty/"

        # Mock S3 client
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_page = {"Contents": []}
        mock_paginator.paginate.return_value = [mock_page]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3_client.return_value = mock_s3

        result = _detect_model_type_from_s3_uri(s3_uri)
        assert result == "unknown"


class TestEnvironmentVariables:
    """Test environment variable handling in model manager."""

    @pytest.mark.unit
    def test_models_dir_environment_variable(self):
        """Test MODELS_DIR environment variable."""
        with patch.dict("os.environ", {"MODELS_DIR": "/custom/models"}):
            import importlib

            import app.model_manager

            importlib.reload(app.model_manager)
            assert app.model_manager.MODELS_DIR == Path("/custom/models")

    @pytest.mark.unit
    def test_llamacpp_dir_environment_variable(self):
        """Test LLAMACPP_DIR environment variable."""
        with patch.dict("os.environ", {"LLAMACPP_DIR": "/custom/llama.cpp"}):
            import importlib

            import app.model_manager

            importlib.reload(app.model_manager)
            assert app.model_manager.LLAMACPP_DIR == Path("/custom/llama.cpp")

    @pytest.mark.unit
    def test_environment_variables_defaults(self):
        """Test default environment variable values."""
        with patch.dict("os.environ", {}, clear=True):
            import importlib

            import app.model_manager

            importlib.reload(app.model_manager)
            assert app.model_manager.MODELS_DIR == Path("/opt/models")
            assert app.model_manager.LLAMACPP_DIR == Path("/opt/llama.cpp")
