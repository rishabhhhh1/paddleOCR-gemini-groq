from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.models.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    make_error,
)
from app.models.solver import ExtractedImage, OcrResult
from app.security.auth import require_backend_api_key
from app.services.consensus_solver import ConsensusSolverError, solve_problem
from app.services.paddle_ocr import OCR_CODE_INVALID_IMAGE, PaddleOcrError, run_ocr_on_images
from app.services.problem_reconstructor import reconstruct_problem
from app.utils.images import ImageValidationError, decode_and_validate_image
from app.utils.logging import (
    Timer,
    log_ocr_failure,
    log_request_event,
    log_solver_failure,
    new_request_id,
)

router = APIRouter(tags=["chat"])


def _client_max_tokens(body: ChatCompletionRequest) -> int | None:
    """Maps the client's optional `max_tokens` onto the solver's output
    budget. None means "use the server's SOLVER_MAX_TOKENS"; each
    provider client applies that setting as a hard ceiling, so a client
    can only ever ask for less than the server allows, never more."""
    if body.max_tokens is None or body.max_tokens <= 0:
        return None
    return body.max_tokens


def _extract_text_and_images(req: ChatCompletionRequest, max_image_mb: int) -> tuple[str, list[ExtractedImage]]:
    """Pulls out a single combined user-facing text prompt and any
    decoded/validated images, preserving message and part order."""
    text_parts: list[str] = []
    images: list[ExtractedImage] = []

    for msg in req.messages:
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            if msg.content.strip():
                text_parts.append(msg.content.strip())
            continue
        for part in msg.content:
            if part.type == "text":
                if part.text.strip():
                    text_parts.append(part.text.strip())
            elif part.type == "image_url":
                decoded = decode_and_validate_image(part.image_url.url, max_image_mb)
                images.append(ExtractedImage(mime_type=decoded.mime_type, data=decoded.data))

    return "\n\n".join(text_parts), images


def _merge_ocr_results(prior: list[OcrResult], fresh: list[OcrResult]) -> list[OcrResult]:
    """Concatenates previously-OCR'd screenshots with this request's own,
    then renumbers 0..N-1 so the reconstruction's SCREENSHOT numbering
    and "N screenshot(s)" count are always contiguous.

    `prior` and `fresh` are ordered independently on purpose: a client's
    earlier /v1/ocr batches carry global indices (4, 5, ...) while fresh
    results restart at 0, so sorting across both would interleave them
    wrongly. Prior batches come first — they were uploaded first."""
    ordered = sorted(prior, key=lambda r: r.index) + sorted(fresh, key=lambda r: r.index)
    return [r.model_copy(update={"index": i}) for i, r in enumerate(ordered)]


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    api_key: str = Depends(require_backend_api_key),
    settings: Settings = Depends(get_settings),
):
    request_id = new_request_id()
    overall_timer = Timer()

    if body.model != settings.PUBLIC_MODEL_NAME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error(
                f"Unknown model '{body.model}'. Use '{settings.PUBLIC_MODEL_NAME}'.",
                "invalid_request_error",
                param="model",
                code="model_not_found",
            ),
        )

    if body.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error(
                "Streaming responses are not yet supported by this backend.",
                "invalid_request_error",
                param="stream",
                code="unsupported_feature",
            ),
        )

    with overall_timer:
        try:
            text, images = _extract_text_and_images(body, settings.MAX_IMAGE_SIZE_MB)
        except ImageValidationError as exc:
            log_request_event(request_id, success=False, http_status=400)
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if exc.code == "request_too_large" else status.HTTP_400_BAD_REQUEST
            raise HTTPException(
                status_code=status_code,
                detail=make_error(exc.message, "invalid_request_error", code=exc.code),
            )

        prior_results = list(body.ocr_results or [])
        total_screenshots = len(prior_results) + len(images)

        # Each limit below treats 0 (or less) as "no limit" — see the
        # convention documented on Settings in app/config.py.
        if settings.MAX_IMAGES > 0 and total_screenshots > settings.MAX_IMAGES:
            log_request_event(request_id, num_images=total_screenshots, success=False, http_status=400)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=make_error(
                    f"Request contains {total_screenshots} screenshots "
                    f"({len(prior_results)} previously OCR'd via /v1/ocr + {len(images)} inline); "
                    f"maximum per solve request is {settings.MAX_IMAGES}.",
                    "invalid_request_error",
                    param="messages",
                    code="too_many_images",
                ),
            )

        if not text and not total_screenshots:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=make_error("Request contains no text or images to solve.", "invalid_request_error"),
            )

        inline_bytes = sum(len(img.data) for img in images)
        if settings.MAX_OCR_BATCH_BYTES > 0 and inline_bytes > settings.MAX_OCR_BATCH_BYTES:
            log_request_event(request_id, num_images=total_screenshots, success=False, http_status=413)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=make_error(
                    f"Inline images total {inline_bytes} bytes; the limit is "
                    f"{settings.MAX_OCR_BATCH_BYTES} (Vercel caps request bodies at "
                    "4.5 MB). Send the screenshots through POST /v1/ocr in batches "
                    "and pass the returned results as `ocr_results` instead.",
                    "invalid_request_error",
                    param="messages",
                    code="request_too_large",
                ),
            )

        ocr_duration_ms = None
        solver_duration_ms = None
        fresh_results: list[OcrResult] = []

        if images:
            # Image path: OCR each screenshot, then solve once. Screenshots
            # already OCR'd through /v1/ocr arrive as `ocr_results` and are
            # merged in below instead of being re-OCRed.
            ocr_timer = Timer()
            try:
                with ocr_timer:
                    fresh_results = await run_ocr_on_images(settings, images)
            except PaddleOcrError as exc:
                log_ocr_failure(request_id, str(exc))
                if exc.code == OCR_CODE_INVALID_IMAGE:
                    # Unreadable screenshot: a bad request, not an
                    # upstream failure (see app/api/ocr.py).
                    log_request_event(request_id, num_images=len(images), success=False, http_status=400)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=make_error(str(exc), "invalid_request_error", param="messages", code=exc.code),
                    )
                log_request_event(request_id, num_images=len(images), success=False, http_status=502)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=make_error(
                        "Local OCR failed to process the screenshots.", "upstream_error", code=exc.code
                    ),
                )
            ocr_duration_ms = ocr_timer.elapsed_ms

        if prior_results or fresh_results:
            reconstructed = reconstruct_problem(
                _merge_ocr_results(prior_results, fresh_results), stitch=body.stitch
            )
            problem_text = reconstructed.text
            if text:
                problem_text = f"{problem_text}\n\nADDITIONAL USER NOTES:\n{text}"
        else:
            # Text-only path: never invoke OCR.
            problem_text = text

        if settings.MAX_PROMPT_CHARS > 0 and len(problem_text) > settings.MAX_PROMPT_CHARS:
            log_request_event(
                request_id,
                num_images=total_screenshots,
                ocr_duration_ms=ocr_duration_ms,
                success=False,
                http_status=413,
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=make_error(
                    f"Assembled prompt is {len(problem_text)} characters, over the "
                    f"{settings.MAX_PROMPT_CHARS} limit — OCR'd screenshots are too verbose "
                    "to solve in one request.",
                    "invalid_request_error",
                    code="prompt_too_large",
                ),
            )

        solver_timer = Timer()
        try:
            with solver_timer:
                solution_text = await solve_problem(
                    settings, problem_text, max_tokens=_client_max_tokens(body)
                )
        except ConsensusSolverError as exc:
            # The detail stays out of the response body (it can name
            # providers and limits), but it has to reach the logs — a
            # truncated answer now surfaces as a 502 here, and without
            # this line the reason for it would be invisible.
            log_solver_failure(request_id, str(exc))
            log_request_event(
                request_id,
                num_images=total_screenshots,
                ocr_duration_ms=ocr_duration_ms,
                success=False,
                http_status=502,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=make_error(
                    "The solver providers failed to produce a response.",
                    "upstream_error",
                    code="solver_failed",
                ),
            )
        solver_duration_ms = solver_timer.elapsed_ms

    response = ChatCompletionResponse(
        model=settings.PUBLIC_MODEL_NAME,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=solution_text),
                finish_reason="stop",
            )
        ],
    )

    log_request_event(
        request_id,
        num_images=total_screenshots,
        ocr_duration_ms=ocr_duration_ms,
        solver_duration_ms=solver_duration_ms,
        total_latency_ms=overall_timer.elapsed_ms,
        success=True,
        http_status=200,
    )

    return response
