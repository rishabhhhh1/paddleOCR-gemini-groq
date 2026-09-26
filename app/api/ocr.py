"""
Two-phase OCR endpoint.

Vercel caps request bodies at 4.5 MB, so 15-20 screenshots as base64
can never arrive in a single `/v1/chat/completions` call. Instead a
client POSTs them to `/v1/ocr` in small batches, keeps the returned
`results`, and sends them all back as the (tiny, text-only) `ocr_results`
field of one solve request.

The server is stateless between the two phases — ordering survives
because each batch carries its own `start_index` and the returned items
are numbered globally, ready to be concatenated or echoed back verbatim.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.models.openai import make_error
from app.models.solver import OcrResult
from app.security.auth import require_backend_api_key
from app.services.paddle_ocr import OCR_CODE_INVALID_IMAGE, PaddleOcrError, run_ocr_on_images
from app.services.problem_reconstructor import format_screenshot_blocks
from app.utils.images import (
    ImageValidationError,
    decode_and_validate_image,
    validate_image_count,
)
from app.utils.logging import Timer, log_ocr_failure, log_request_event, new_request_id

router = APIRouter(tags=["ocr"])


class OcrBatchRequest(BaseModel):
    # Base64 data URLs in screenshot order, e.g.
    # "data:image/png;base64,iVBOR..."
    images: list[str] = Field(min_length=1)
    # Global position of images[0]. Use the `next_start_index` returned
    # by the previous batch so numbering stays continuous across calls.
    start_index: int = 0
    # Only affects this response's convenience `text` field (see below) —
    # set it to match whatever you will later pass as `stitch` to
    # /v1/chat/completions, so a client that concatenates `text` across
    # batches instead of echoing `results` sees the same reconstruction
    # the solve step will actually use.
    stitch: bool = True


class OcrBatchResponse(BaseModel):
    object: str = "ocr.result"
    start_index: int
    count: int
    next_start_index: int
    results: list[OcrResult]
    # The same results as numbered SCREENSHOT blocks with no preamble —
    # concatenate `text` across batches to get one ordered document.
    text: str


@router.post("/v1/ocr", response_model=OcrBatchResponse)
async def ocr_batch(
    body: OcrBatchRequest,
    api_key: str = Depends(require_backend_api_key),
    settings: Settings = Depends(get_settings),
):
    request_id = new_request_id()

    if body.start_index < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error("start_index must be >= 0.", "invalid_request_error", param="start_index"),
        )

    try:
        images = [decode_and_validate_image(url, settings.MAX_IMAGE_SIZE_MB) for url in body.images]
    except ImageValidationError as exc:
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if exc.code == "request_too_large" else status.HTTP_400_BAD_REQUEST
        log_request_event(request_id, success=False, http_status=status_code)
        raise HTTPException(
            status_code=status_code,
            detail=make_error(exc.message, "invalid_request_error", code=exc.code),
        )

    try:
        validate_image_count(len(images), settings.MAX_OCR_BATCH_IMAGES)
    except ImageValidationError as exc:
        log_request_event(request_id, num_images=len(images), success=False, http_status=400)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error(exc.message, "invalid_request_error", code=exc.code),
        )

    total_bytes = sum(len(img.data) for img in images)
    # MAX_OCR_BATCH_BYTES of 0 (or less) means "no limit" — see the
    # convention documented on Settings in app/config.py.
    if settings.MAX_OCR_BATCH_BYTES > 0 and total_bytes > settings.MAX_OCR_BATCH_BYTES:
        # Checked here rather than left to the platform: past Vercel's
        # 4.5 MB body cap the request dies at the edge with a bare 413
        # and never reaches this handler, so the client would get no
        # hint that it should simply send fewer images per batch.
        log_request_event(request_id, num_images=len(images), success=False, http_status=413)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=make_error(
                f"Batch holds {total_bytes} bytes of image data; the limit is "
                f"{settings.MAX_OCR_BATCH_BYTES}. Split it into smaller batches "
                "or upload the screenshots at a lower resolution.",
                "invalid_request_error",
                param="images",
                code="request_too_large",
            ),
        )

    ocr_timer = Timer()
    try:
        with ocr_timer:
            raw_results = await run_ocr_on_images(settings, images)
    except PaddleOcrError as exc:
        log_ocr_failure(request_id, str(exc))
        if exc.code == OCR_CODE_INVALID_IMAGE:
            # The screenshot itself is unreadable (truncated file, no
            # pixels, a format Pillow only half-parses). Validation
            # passes those because verify() does not fully decode, so
            # this is where they surface — and it is a bad request, not
            # an upstream failure, so it must not masquerade as a 502.
            log_request_event(request_id, num_images=len(images), success=False, http_status=400)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=make_error(str(exc), "invalid_request_error", param="images", code=exc.code),
            )
        log_request_event(request_id, num_images=len(images), success=False, http_status=502)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=make_error(
                "Local OCR failed to process the screenshots.", "upstream_error", code=exc.code
            ),
        )

    offset = body.start_index
    results = [
        OcrResult(index=offset + r.index, text=r.text, unreadable=r.unreadable)
        for r in raw_results
    ]

    log_request_event(
        request_id,
        num_images=len(images),
        ocr_duration_ms=ocr_timer.elapsed_ms,
        total_latency_ms=ocr_timer.elapsed_ms,
        success=True,
        http_status=200,
    )

    return OcrBatchResponse(
        start_index=offset,
        count=len(results),
        next_start_index=offset + len(results),
        results=results,
        text=format_screenshot_blocks(results, stitch=body.stitch),
    )
