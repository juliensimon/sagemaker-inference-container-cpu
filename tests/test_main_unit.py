"""
Unit tests for the main application module.

This module contains unit tests for all functions in app.main.py,
including request handling, path selection, and proxy functionality.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import (
    _choose_openai_path,
    _proxy_request,
    app,
    invocations,
    lifespan,
    openai_passthrough,
    ping,
    spawn_llama_server,
)


class TestChooseOpenAIPath:
    """Test the _choose_openai_path function"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_choose_path_with_messages(self):
        """Test path selection when messages field is present."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = await _choose_openai_path(body)
        assert result == "/v1/chat/completions"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_choose_path_without_messages(self):
        """Test path selection when messages field is not present."""
        body = {"prompt": "Hello", "max_tokens": 100}
        result = await _choose_openai_path(body)
        assert result == "/v1/completions"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_choose_path_empty_body(self):
        """Test path selection with empty body."""
        body = {}
        result = await _choose_openai_path(body)
        assert result == "/v1/completions"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_choose_path_with_other_fields(self):
        """Test path selection with other fields but no messages."""
        body = {"temperature": 0.7, "top_p": 0.9}
        result = await _choose_openai_path(body)
        assert result == "/v1/completions"


# TestSpawnLlamaServer class removed - functionality tested in live container tests


class TestProxyRequest:
    """Test the _proxy_request function"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_proxy_request_non_streaming(self, mock_client_class):
        """Test non-streaming proxy request."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"result": "test"}'
        mock_client.request.return_value = mock_response

        mock_request = AsyncMock()
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b"test body")
        mock_request.headers = {"content-type": "application/json"}

        response = await _proxy_request(mock_request, "/test", False)

        assert response.status_code == 200
        mock_client.request.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_proxy_request_streaming(self, mock_client_class):
        """Test streaming proxy request."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.aiter_bytes.return_value = [b"data: test\n"]
        mock_client.stream.return_value.__aenter__.return_value = mock_response

        mock_request = AsyncMock()
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b"test body")
        mock_request.headers = {"content-type": "application/json"}

        response = await _proxy_request(mock_request, "/test", True)

        assert response.status_code == 200
        assert response.media_type == "text/event-stream"

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_proxy_request_headers_cleanup(self, mock_client_class):
        """Test that headers are properly cleaned up."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"result": "test"}'
        mock_client.request.return_value = mock_response

        mock_request = AsyncMock()
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b"test body")
        mock_request.headers = {
            "content-type": "application/json",
            "authorization": "Bearer token",
            "host": "localhost:8080",
        }

        await _proxy_request(mock_request, "/test", False)

        # Verify headers were cleaned up
        call_args = mock_client.request.call_args
        headers = call_args[1]["headers"]
        assert "host" not in headers
        assert "authorization" in headers

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_streaming_response_format(self, mock_client_class):
        """Test that streaming responses are properly formatted."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.aiter_bytes.return_value = [b"data: test\n"]
        mock_client.stream.return_value.__aenter__.return_value = mock_response

        mock_request = AsyncMock()
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b"test body")
        mock_request.headers = {"content-type": "application/json"}

        response = await _proxy_request(mock_request, "/v1/chat/completions", True)

        # Verify response format
        assert response.status_code == 200
        assert response.media_type == "text/event-stream"

    # Streaming lifecycle tests removed - replaced with live container tests


# TestFastAPIEndpoints class removed - functionality tested in live container tests


class TestLifespanManager:
    """Test the lifespan manager"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("app.main.spawn_llama_server")
    @patch("app.main.prepare_model_and_get_path")
    async def test_lifespan_startup(self, mock_prepare, mock_spawn):
        """Test lifespan startup."""
        mock_prepare.return_value = "/path/to/model.gguf"
        mock_spawn.return_value = None

        async with lifespan(app):
            mock_prepare.assert_called_once()
            mock_spawn.assert_called_once_with("/path/to/model.gguf")

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("app.main.spawn_llama_server")
    @patch("app.main.prepare_model_and_get_path")
    async def test_lifespan_shutdown(self, mock_prepare, mock_spawn):
        """Test lifespan shutdown."""
        mock_prepare.return_value = "/path/to/model.gguf"
        mock_spawn.return_value = None

        async with lifespan(app):
            pass  # Shutdown happens automatically


class TestEnvironmentVariables:
    """Test environment variable handling"""

    @pytest.mark.unit
    def test_default_port_configuration(self):
        """Test default port configuration."""
        with patch.dict("os.environ", {}, clear=True):
            from app.main import APP_PORT, UPSTREAM_HOST, UPSTREAM_PORT

            assert APP_PORT == 8080
            assert UPSTREAM_PORT == 8081
            assert UPSTREAM_HOST == "127.0.0.1"

    @pytest.mark.unit
    def test_custom_port_configuration(self):
        """Test custom port configuration."""
        with patch.dict(
            "os.environ",
            {"PORT": "9000", "UPSTREAM_PORT": "9001", "UPSTREAM_HOST": "0.0.0.0"},
        ):
            # Reload the module to get new environment values
            import importlib

            import app.main

            importlib.reload(app.main)

            assert app.main.APP_PORT == 9000
            assert app.main.UPSTREAM_PORT == 9001
            assert app.main.UPSTREAM_HOST == "0.0.0.0"
