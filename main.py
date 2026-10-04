"""
URL Reader Tools API.

Provides LLM tools for fetch web pages (URLs) and convert them to LLM-friendly formats like markdown, text.

Author: bgeneto
Date: 2025-05-02
Version: 1.0.4
Last Modified: 2025-07-02
"""

import base64
import os
import time
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

# Prefix every log line so it is easy to follow one screenshot end-to-end:
#   docker compose logs -f url-reader-server | grep '\[shot\]'
LOG_TAG = "[shot]"

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


# Rendering a page with headless Chrome routinely takes longer than httpx's
# 5-second default. Keep this above the reader's own crawl timeout (30s default).
SCREENSHOT_TIMEOUT = float(os.getenv("SCREENSHOT_TIMEOUT", "90"))
# Ask the reader for more time; it caps X-Timeout at 180 seconds anyway.
CRAWL_TIMEOUT_SECONDS = int(os.getenv("CRAWL_TIMEOUT_SECONDS", "60"))

# Lightweight counters so GET /debug/status can show recent activity at a glance.
_stats = {"requests": 0, "ok": 0, "failed": 0, "last_error": None, "last_url": None}


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


async def _fetch_screenshot_bytes(url: str, respond_with: str, params: dict = None) -> bytes:
    """Ask the reader for a screenshot and return the PNG bytes.

    Kept separate from _fetch_content so failures carry the upstream status and
    body instead of httpx's generic "Server error '500 ...'" text.
    """
    decoded_url = unquote(url)
    target_url = f"{READER_BASE_URL}/{decoded_url}"
    # X-Timeout is capped at 180s by the reader; headers win over query params.
    crawl_timeout = min(max(CRAWL_TIMEOUT_SECONDS, 1), 180)
    headers = {
        "X-Respond-With": respond_with,
        "X-Timeout": str(crawl_timeout),
    }

    logger.info(
        "%s request: respond_with=%s url=%s params=%s "
        "(client timeout=%ss, X-Timeout=%ss)",
        LOG_TAG,
        respond_with,
        decoded_url,
        params or {},
        SCREENSHOT_TIMEOUT,
        crawl_timeout,
    )

    _stats["requests"] += 1
    _stats["last_url"] = decoded_url
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=SCREENSHOT_TIMEOUT) as client:
        try:
            response = await client.get(
                target_url, headers=headers, params=params, follow_redirects=True
            )
            elapsed = time.monotonic() - started

            # The reader answers a screenshot request with 302 -> /instant-screenshots/<uuid>.png,
            # so a non-empty history proves the PNG redirect was followed successfully.
            hops = len(response.history)
            if hops:
                logger.info(
                    "%s followed %d redirect(s), landed on %s",
                    LOG_TAG,
                    hops,
                    response.url,
                )

            logger.info(
                "%s upstream: status=%s content_type=%s bytes=%s elapsed=%.2fs",
                LOG_TAG,
                response.status_code,
                response.headers.get("content-type", "(none)"),
                len(response.content),
                elapsed,
            )

            response.raise_for_status()

            if not hops:
                logger.warning(
                    "%s no redirect seen; body may be an error page rather than a PNG "
                    "(starts with %r)",
                    LOG_TAG,
                    response.content[:32],
                )

            mime = _sniff_image_mime(response.content, fallback="")
            if not mime:
                logger.warning(
                    "%s body is not a recognised image format (magic bytes %r); "
                    "returning it as PNG anyway",
                    LOG_TAG,
                    response.content[:8],
                )
            _stats["ok"] += 1
            return response.content

        except httpx.HTTPStatusError as e:
            elapsed = time.monotonic() - started
            status = e.response.status_code if e.response is not None else "?"
            body = e.response.text[:500] if e.response is not None else str(e)
            logger.error(
                "%s FAILED upstream status=%s after %.2fs for %s: %s",
                LOG_TAG,
                status,
                elapsed,
                decoded_url,
                body,
            )
            hint = ""
            if status >= 500:
                hint = (
                    " The reader's headless Chrome failed to render the page "
                    "(see `docker compose logs url-reader-app`); it is usually a "
                    "render timeout, a crashed browser, or a blocked page."
                )
            _stats["failed"] += 1
            _stats["last_error"] = f"upstream {status}: {body[:200]}"
            raise HTTPException(
                status_code=502,
                detail=f"Reader backend returned {status} for {decoded_url}: {body}{hint}",
            )
        except httpx.TimeoutException as e:
            elapsed = time.monotonic() - started
            logger.error(
                "%s TIMEOUT after %.2fs (limit %ss) for %s: %s",
                LOG_TAG,
                elapsed,
                SCREENSHOT_TIMEOUT,
                decoded_url,
                e,
            )
            _stats["failed"] += 1
            _stats["last_error"] = f"timeout after {SCREENSHOT_TIMEOUT:.0f}s: {e}"
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Reader backend did not answer within {SCREENSHOT_TIMEOUT:.0f}s "
                    f"while rendering {decoded_url}. Script-heavy pages may need a "
                    f"longer SCREENSHOT_TIMEOUT or a smaller viewport."
                ),
            )
        except httpx.HTTPError as e:
            elapsed = time.monotonic() - started
            logger.error(
                "%s TRANSPORT ERROR after %.2fs for %s: %s",
                LOG_TAG,
                elapsed,
                decoded_url,
                e,
            )
            _stats["failed"] += 1
            _stats["last_error"] = f"transport error: {e}"
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach the reader backend for {decoded_url}: {e}",
            )


# Screenshot endpoint with dimensions
@app.get(
    "/screenshot/{url:path}",
    summary="Fetch screenshot of a URL",
    operation_id="get_screenshot",
    description=(
        "Returns a screenshot of the given URL as a single image data URI. "
        "The optional `debug` parameter only affects diagnostics: pass 2 to get a "
        "JSON error report instead of an HTTP error when the page cannot be "
        "rendered, which is useful for troubleshooting."
    ),
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
    debug: int = Query(
        0,
        description=(
            "Diagnostics for this same request. 1 = always return a JSON debug "
            "summary; 2 = only when the request failed. Useful from a browser."
        ),
    ),
):
    try:
        data = await _fetch_screenshot_bytes(
            url, "screenshot", {"width": width, "height": height}
        )
    except HTTPException as e:
        if debug >= 2:
            return JSONResponse(
                status_code=200,
                content={
                    "ok": False,
                    "url": unquote(url),
                    "status": e.status_code,
                    "error": e.detail,
                },
            )
        raise

    data_uri = _as_image_data_uri(data)

    if debug >= 1:
        logger.info(
            "%s debug summary requested for %s", LOG_TAG, unquote(url)
        )
        return JSONResponse(
            content={
                "ok": True,
                "url": unquote(url),
                "mime": _sniff_image_mime(data, fallback="(unknown)"),
                "image_bytes": len(data),
                "data_uri_chars": len(data_uri),
                # Open WebUI turns the tool result into a chat attachment only when
                # the whole result is one image data URI (see README).
                "openwebui_data_uri_ok": data_uri.startswith("data:image/"),
                "data_uri_preview": data_uri[:64] + "...",
            }
        )

    logger.info(
        "screenshot %s -> %d bytes, data URI %d chars", url, len(data), len(data_uri)
    )
    # Returned as one whole data URI string, not wrapped in JSON.
    return PlainTextResponse(content=data_uri, media_type="text/plain")


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
    data = await _fetch_screenshot_bytes(url, "pageshot")
    data_uri = _as_image_data_uri(data)
    logger.info(
        "pageshot %s -> %d bytes, data URI %d chars", url, len(data), len(data_uri)
    )
    return PlainTextResponse(content=data_uri, media_type="text/plain")


@app.get(
    "/debug/status",
    summary="Diagnostics: configuration and reachability of the reader backend.",
)
async def debug_status():
    """Report the effective config, recent counters, and reader reachability.

    Open this in a browser, e.g. http://<docker-host>:8000/debug/status
    """
    logger.info("%s running self-check against %s", LOG_TAG, READER_BASE_URL)
    result = {
        "reader_base_url": READER_BASE_URL,
        "client_timeout_s": SCREENSHOT_TIMEOUT,
        "crawl_timeout_s": CRAWL_TIMEOUT_SECONDS,
        "webui_needs": "tool result must be one whole image data URI",
        "stats": dict(_stats),
    }

    started = time.monotonic()
    try:
        # HEAD on the reader root: enough to prove reachability, no page rendered.
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                "HEAD", f"{READER_BASE_URL}/", follow_redirects=False
            )
        result.update(
            {
                "reader_reachable": True,
                "reader_http_status": response.status_code,
                "elapsed_s": round(time.monotonic() - started, 2),
            }
        )
    except httpx.HTTPError as e:
        result.update(
            {
                "reader_reachable": False,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.monotonic() - started, 2),
            }
        )
        logger.error("%s self-check could not reach reader: %s", LOG_TAG, e)

    logger.info("%s self-check result: %s", LOG_TAG, result)
    return JSONResponse(content=result)


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
            "debug_status": "/debug/status",
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
