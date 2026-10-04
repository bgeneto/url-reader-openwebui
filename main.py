"""
URL Reader Tools API.

Provides LLM tools for fetch web pages (URLs) and convert them to LLM-friendly formats like markdown, text.

Author: bgeneto
Date: 2025-05-02
Version: 1.0.4
Last Modified: 2025-07-02
"""

import base64
import traceback
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
from urllib.parse import unquote

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("reader-tools")

app = FastAPI(
    title="URL Reader",
    description="Provides LLM tools for fetch web pages (URLs) and convert them to LLM-friendly formats like markdown, text and others like screenshots.",
    version="1.0.4",
)

# Enable CORS (allowing all origins for simplicity)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated base URL for the Reader tool
READER_BASE_URL = "http://url-reader-app:3000"

# Supported response types
RESPONSE_TYPES = {
    "markdown": "markdown",
    "html": "html",
    "text": "text",
    "screenshot": "screenshot",
    "pageshot": "pageshot",
}


def _sniff_image_mime(data: bytes, fallback: str = "image/png") -> str:
    """Return the MIME type of an image from its magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def _as_image_data_uri(data: bytes, content_type: str = "") -> str:
    """Encode image bytes as a single ``data:<mime>;base64,<payload>`` URI string.

    Why this matters for Open WebUI tool calling: when a tool result is one whole
    image data URI, Open WebUI moves it out of the model context into a chat
    attachment and shows it inline. A base64 blob nested inside a JSON object
    (the previous shape of this endpoint) is instead serialised into the tool
    message as text, which can cost hundreds of thousands of tokens and leaves
    the model unable to render it.
    """
    mime = content_type.split(";")[0].strip()
    if not mime.startswith("image/"):
        mime = _sniff_image_mime(data)
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


# Helper to fetch and wrap responses
async def _fetch_content(url: str, respond_with: str, client_params: dict = None):
    # URL decode the incoming URL
    decoded_url = unquote(url)
    target_url = f"{READER_BASE_URL}/{decoded_url}"
    headers = {"X-Respond-With": respond_with}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                target_url, headers=headers, params=client_params, follow_redirects=True
            )
            response.raise_for_status()
            content_type = f"text/{respond_with}"
            if "application/json" in content_type:
                return JSONResponse(content=response.json())
            elif (
                "text/html" in content_type
                or "text/plain" in content_type
                or "text/markdown" in content_type
            ):
                return JSONResponse(
                    content={"content": response.text, "content_type": content_type}
                )
            else:
                import base64

                return JSONResponse(
                    content={
                        "content_base64": base64.b64encode(response.content).decode(
                            "utf-8"
                        ),
                        "content_type": "image/png;base64",
                    }
                )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


# Markdown endpoint
@app.get(
    "/markdown/{url:path}",
    summary="Fetch markdown of a URL",
    operation_id="get_markdown",
    responses={
        200: {
            "description": "Markdown content in JSON",
            "content": {
                "application/json": {
                    "example": {"content": "# Heading", "content_type": "text/markdown"}
                }
            },
        }
    },
)
async def get_markdown(url: str):
    return await _fetch_content(url, "markdown")


# HTML endpoint
@app.get(
    "/html/{url:path}",
    summary="Fetch HTML of a URL",
    operation_id="get_html",
    responses={
        200: {
            "description": "HTML content in JSON",
            "content": {
                "application/json": {
                    "example": {
                        "content": "<html>...</html>",
                        "content_type": "text/html",
                    }
                }
            },
        }
    },
)
async def get_html(url: str):
    return await _fetch_content(url, "html")


# Text endpoint
@app.get(
    "/text/{url:path}",
    summary="Fetch plain text of a URL",
    operation_id="get_text",
    responses={
        200: {
            "description": "Text content in JSON",
            "content": {
                "application/json": {
                    "example": {
                        "content": "Plain text...",
                        "content_type": "text/plain",
                    }
                }
            },
        }
    },
)
async def get_text(url: str):
    return await _fetch_content(url, "plain")


# Screenshot endpoint with dimensions
@app.get(
    "/screenshot/{url:path}",
    summary="Fetch screenshot of a URL",
    operation_id="get_screenshot",
    responses={
        200: {
            "description": (
                "Screenshot as a single image data URI string "
                "(data:image/png;base64,...), ready to be attached by the client."
            ),
            "content": {
                "text/plain": {
                    "example": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
                }
            },
        }
    },
)
async def get_screenshot(
    url: str,
    width: int = Query(1280, description="Viewport width"),
    height: int = Query(720, description="Viewport height"),
):
    target_url = f"{READER_BASE_URL}/{url}"
    headers = {"X-Respond-With": "screenshot"}
    params = {"width": width, "height": height}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                target_url, headers=headers, params=params, follow_redirects=True
            )
            response.raise_for_status()
            data_uri = _as_image_data_uri(
                response.content, response.headers.get("content-type", "")
            )
            logger.info(
                "screenshot %s -> %d bytes, data URI %d chars",
                url,
                len(response.content),
                len(data_uri),
            )
            # Returned as one whole data URI string, not wrapped in JSON.
            return PlainTextResponse(content=data_uri, media_type="text/plain")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


# Pageshot endpoint
@app.get(
    "/pageshot/{url:path}",
    summary="Fetch full page screenshot of a URL",
    operation_id="get_pageshot",
    responses={
        200: {
            "description": (
                "Full page screenshot as a single image data URI string "
                "(data:image/png;base64,...), ready to be attached by the client."
            ),
            "content": {
                "text/plain": {
                    "example": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
                }
            },
        }
    },
)
async def get_pageshot(url: str):
    target_url = f"{READER_BASE_URL}/{url}"
    headers = {"X-Respond-With": "pageshot"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                target_url, headers=headers, follow_redirects=True
            )
            response.raise_for_status()
            data_uri = _as_image_data_uri(
                response.content, response.headers.get("content-type", "")
            )
            logger.info(
                "pageshot %s -> %d bytes, data URI %d chars",
                url,
                len(response.content),
                len(data_uri),
            )
            return PlainTextResponse(content=data_uri, media_type="text/plain")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


@app.get(
    "/",
    summary="Retrieve URL content in LLM-friendly formats like markdown, text and others like screenshots.",
)
async def root():
    return {
        "message": "Reader Tools API is running.",
        "endpoints": {
            "markdown": "/markdown/{url}",
            "html": "/html/{url}",
            "text": "/text/{url}",
            "screenshot": "/screenshot/{url}",
            "pageshot": "/pageshot/{url}",
        },
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"Internal server error: {str(exc)}"},
    )
