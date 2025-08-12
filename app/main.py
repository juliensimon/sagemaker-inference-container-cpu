"""
SageMaker inference container main application.

This module provides a FastAPI-based inference server that acts as a proxy between
SageMaker and the llama.cpp server. It handles model preparation, server spawning,
and request routing for both OpenAI-compatible API endpoints and SageMaker invocations.

The application supports:
- Automatic model downloading from HuggingFace or S3
- Model conversion from HuggingFace format to GGUF
- Model quantization
- Streaming and non-streaming inference
- OpenAI-compatible API endpoints
- SageMaker invocations endpoint

Environment Variables:
    PORT: Port for the main application (default: 8080)
    UPSTREAM_PORT: Port for the llama.cpp server (default: 8081)
    UPSTREAM_HOST: Host for the llama.cpp server (default: 127.0.0.1)
    LLAMA_CPP_ARGS: Additional arguments for llama-server
"""

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
from typing import Any, Dict, List, Optional

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.model_manager import prepare_model_and_get_path

APP_PORT = int(os.getenv("PORT", "8080"))
UPSTREAM_PORT = int(os.getenv("UPSTREAM_PORT", "8081"))
UPSTREAM_HOST = os.getenv("UPSTREAM_HOST", "127.0.0.1")

# Global process handle
llama_proc: Optional[subprocess.Popen] = None

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager for application startup and shutdown.

    Handles model preparation and llama-server process lifecycle.

    Args:
        app: FastAPI application instance

    Yields:
        None: Application runs during yield

    Note:
        On startup: Prepares model and spawns llama-server
        On shutdown: Terminates llama-server process and all child processes
    """
    # Startup
    model_path = await asyncio.get_event_loop().run_in_executor(
        None, prepare_model_and_get_path
    )
    await spawn_llama_server(model_path)

    yield

    # Shutdown
    global llama_proc
    if llama_proc and llama_proc.poll() is None:
        try:
            parent = psutil.Process(llama_proc.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            try:
                parent.wait(timeout=10)
            except psutil.TimeoutExpired:
                parent.kill()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


async def spawn_llama_server(model_path: Optional[str]) -> None:
    """
    Spawn the llama.cpp server process.

    Args:
        model_path: Path to the model file to load

    Raises:
        RuntimeError: If llama-server binary is not found or no model path provided
    """
    global llama_proc
    if llama_proc and llama_proc.poll() is None:
        return

    llama_server_bin = shutil.which("llama-server")
    if not llama_server_bin:
        raise RuntimeError("llama-server binary not found in PATH")

    llama_args = os.getenv("LLAMA_CPP_ARGS", "")
    extra_args = shlex.split(llama_args) if llama_args else []

    # Base server args (OpenAI API is default)
    base_args: List[str] = [
        llama_server_bin,
        "--host",
        "127.0.0.1",
        "--port",
        str(UPSTREAM_PORT),
    ]

    # Model source: always use local model path
    if not model_path:
        raise RuntimeError("No model path provided")
    model_args = ["--model", model_path]

    # Avoid duplicate flags from user
    normalized = " ".join(extra_args)

    # Don't add duplicate host/port since they're already in base_args
    # The user's extra_args will override base_args if they specify host/port

    cmd = base_args + model_args + extra_args

    env = os.environ.copy()

    llama_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    async def _log_output():
        """Log output from the llama-server process."""
        assert llama_proc and llama_proc.stdout
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, llama_proc.stdout.readline)
            if not line:
                break
            print(f"[llama-server] {line}", end="")

    asyncio.create_task(_log_output())


@app.get("/ping")
async def ping():
    """
    Health check endpoint.

    Returns:
        PlainTextResponse: "OK" with 200 status code
    """
    return PlainTextResponse("OK", status_code=200)


async def _choose_openai_path(body: Dict[str, Any]) -> str:
    """
    Determine the appropriate OpenAI API endpoint based on request body.

    Args:
        body: Request body dictionary

    Returns:
        API endpoint path (/v1/chat/completions or /v1/completions)
    """
    if "messages" in body:
        return "/v1/chat/completions"
    return "/v1/completions"


async def _proxy_request(request: Request, path: str, stream: bool) -> Response:
    """
    Proxy a request to the upstream llama.cpp server.

    Args:
        request: FastAPI request object
        path: Target path on the upstream server
        stream: Whether this is a streaming request

    Returns:
        Response: Proxied response from upstream server
    """
    upstream = f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("connection", None)

    timeout = httpx.Timeout(120.0, connect=30.0)

    if stream:
        req_stream = await request.body()

        # For real streaming, we need to create a client that stays alive
        async def stream_response():
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    request.method, upstream, content=req_stream, headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_response(), status_code=200, media_type="text/event-stream"
        )
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            data = await request.body()
            r = await client.request(
                request.method, upstream, content=data, headers=headers
            )
            safe_headers = {}
            ct = r.headers.get("content-type")
            if ct:
                safe_headers["content-type"] = ct
            rid = r.headers.get("x-request-id")
            if rid:
                safe_headers["x-request-id"] = rid
            return Response(
                content=r.content, status_code=r.status_code, headers=safe_headers
            )


@app.post("/invocations")
async def invocations(request: Request):
    """
    SageMaker invocations endpoint.

    Handles inference requests from SageMaker and routes them to the appropriate
    OpenAI-compatible endpoint on the llama.cpp server.

    Args:
        request: FastAPI request object containing the inference payload

    Returns:
        Response: Inference response from the model

    Raises:
        HTTPException: If request body is invalid JSON
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    stream = bool(body.get("stream", False))
    path = await _choose_openai_path(body)
    return await _proxy_request(request, path, stream)


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH"])
async def openai_passthrough(full_path: str, request: Request):
    """
    OpenAI-compatible API passthrough endpoint.

    Routes requests to the upstream llama.cpp server while providing basic validation
    for common endpoints like chat/completions and completions.

    Args:
        full_path: Full API path (e.g., "chat/completions", "models", etc.)
        request: FastAPI request object

    Returns:
        Response: Proxied response from upstream server

    Raises:
        HTTPException: If required fields are missing or JSON is invalid
    """
    stream = False
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.json()
            stream = bool(body.get("stream", False))

            # Basic validation for required fields
            if full_path == "chat/completions":
                if "messages" not in body:
                    raise HTTPException(
                        status_code=400, detail="Missing required field: messages"
                    )
                if not body["messages"] or not isinstance(body["messages"], list):
                    raise HTTPException(
                        status_code=400, detail="Invalid messages field"
                    )
            elif full_path == "completions":
                if "prompt" not in body:
                    raise HTTPException(
                        status_code=400, detail="Missing required field: prompt"
                    )

        except HTTPException:
            raise
        except Exception:
            # Check if this is a streaming request by headers
            if request.headers.get("accept", "").startswith("text/event-stream"):
                stream = True
            else:
                # Return 400 for invalid JSON in non-streaming requests
                raise HTTPException(status_code=400, detail="Invalid JSON body")
    elif request.method == "GET":
        stream = request.headers.get("accept", "").startswith("text/event-stream")

    return await _proxy_request(request, f"/v1/{full_path}", stream)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")
