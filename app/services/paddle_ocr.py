"""
Local PaddleOCR integration.

Unlike the Mistral OCR backend this replaces, PaddleOCR runs fully
in-process on this server (CPU or GPU) — there is no external API
call and no API key. Each screenshot is decoded to a numpy array in
memory (never written to disk), normalized so its background color
never decides how much text survives, and run through PaddleOCR's
detection + recognition pipeline. The recognized text lines are then
sorted into top-to-bottom, left-to-right reading order (PaddleOCR
returns them in detection-confidence/internal order, not reading
order) and joined into one text block per screenshot, mirroring the
shape `problem_reconstructor.py` expects from the old Mistral-backed
service.

Two things fall out of "runs locally" that are worth knowing about:

1. Model loading is expensive (multiple seconds, and the weights —
   tens to a few hundred MB depending on PADDLE_OCR_LANG/model
   variant — are downloaded on first use and cached on disk under
   `~/.paddlex` or `PADDLE_PDX_CACHE_HOME`). A small pool of
   already-loaded engine instances is created once at process startup
   (see `warm_up`, called from `app/main.py`) and reused for the life
   of the process, instead of paying that cost per request.
2. `.predict()` is synchronous, CPU/GPU-bound code, so every call is
   dispatched to a worker thread via `asyncio.to_thread` rather than
   blocking the event loop.

Unlike a hosted OCR API, PaddleOCR does not produce Markdown — no
headings, tables, or math notation are reconstructed, only plain text
lines. For DSA problem screenshots (mostly prose, code blocks, and
plain constraint lists) this is normally enough, but it's a real
capability drop from the Mistral OCR backend for anything
table-or-formula-heavy.
"""
from __future__ import annotations

import asyncio
import io
import threading
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.config import Settings
from app.models.solver import ExtractedImage, OcrResult


# Diagnostic codes carried on PaddleOcrError. They reach the client as the
# OpenAI-style `error.code` field, because the message body is deliberately
# generic ("Local OCR failed to process the screenshots.") and that left a
# 502 completely unattributable — an unreadable screenshot, a model that
# never loaded, and an out-of-memory inference all looked identical from
# outside. None of them name a path, a provider, or a limit.
OCR_CODE_INVALID_IMAGE = "invalid_image"  # we could not decode the file: the client's problem
OCR_CODE_ENGINE_LOAD = "ocr_engine_load_failed"
OCR_CODE_INFERENCE = "ocr_inference_failed"


class PaddleOcrError(Exception):
    """Raised when local PaddleOCR inference fails."""

    def __init__(self, message: str, *, code: str = OCR_CODE_INFERENCE):
        super().__init__(message)
        self.code = code


class _PooledEngine:
    """One long-lived PaddleOCR pipeline instance plus a lock.

    A single PaddleOCR object is not documented as safe to call from
    multiple threads concurrently, so each instance in the pool is
    only ever used by one thread at a time; running more than one
    screenshot truly concurrently means having more than one instance
    (see OCR_MAX_CONCURRENCY), at the cost of that many copies of the
    model weights in memory.
    """

    def __init__(self, settings: Settings):
        # Imported lazily: paddleocr/paddlepaddle are heavy optional
        # dependencies. Importing them at module load time would (a)
        # slow down every process that merely imports this module,
        # including tests, which mock this service out entirely and
        # never need the real engine, and (b) make it impossible to
        # import app.services.paddle_ocr at all in an environment
        # where paddleocr/paddlepaddle haven't been installed yet.
        from paddleocr import PaddleOCR

        self.lock = threading.Lock()
        kwargs: dict[str, Any] = {
            "device": settings.PADDLE_OCR_DEVICE,
            "use_textline_orientation": settings.PADDLE_OCR_USE_TEXTLINE_ORIENTATION,
            "use_doc_orientation_classify": settings.PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY,
            "use_doc_unwarping": settings.PADDLE_OCR_USE_DOC_UNWARPING,
            "text_det_limit_type": settings.PADDLE_OCR_DET_LIMIT_TYPE,
            "text_det_limit_side_len": settings.PADDLE_OCR_DET_LIMIT_SIDE_LEN,
        }
        # paddleocr ignores `lang` as soon as ANY model name is set, so
        # the two names travel together. Set PADDLE_OCR_DET_MODEL=lang to
        # hand model selection back to PADDLE_OCR_LANG (an empty value
        # can't be expressed through the env, since blank vars are
        # treated as unset — see config.py).
        det_model = settings.PADDLE_OCR_DET_MODEL.strip().lower()
        if det_model and det_model != "lang":
            kwargs["text_detection_model_name"] = settings.PADDLE_OCR_DET_MODEL
            kwargs["text_recognition_model_name"] = settings.PADDLE_OCR_REC_MODEL
        else:
            kwargs["lang"] = settings.PADDLE_OCR_LANG
        self.engine = PaddleOCR(**kwargs)


_engine_pool: list[_PooledEngine] | None = None
_pool_lock = threading.Lock()

# Screenshots (especially phone photos) can be far larger than PaddleOCR
# needs — its detection stage resizes internally anyway. Capping the
# longest side keeps a multi-image request from holding a pile of
# full-resolution RGB arrays in memory at once, which matters a lot on
# hosts with a hard memory ceiling (e.g. Vercel Hobby's 2 GB).
MAX_IMAGE_SIDE_PX = 2000
# Tall screenshots (full-page captures, a whole problem thread in one
# image) must NOT be shrunk to MAX_IMAGE_SIDE_PX — a 900x14000 capture
# squeezed into 2000px puts text at 1/7 scale and OCR returns gibberish.
# Instead they are sliced into tiles of this height, each OCR'd at (near)
# native resolution, and the lines concatenated. Slices are cut at blank
# rows so no text line is ever split across two tiles.
TILE_HEIGHT_PX = 1800
# Screenshots narrower than this are upscaled before OCR: a 480px-wide
# phone screenshot has text too small for the recognition model.
MIN_TEXT_WIDTH_PX = 960
# Never cut a tile thinner than this (a cut lands in the tail of the
# image instead of dropping a sliver), and how far around each target
# row to look for a blank row to cut at.
MIN_SEGMENT_PX = 400
SPLIT_SEARCH_PX = 500


def _scale_for_width(width: int) -> float:
    """Output/source pixel ratio: enlarge narrow screenshots so text stays
    legible, but never past MAX_IMAGE_SIDE_PX in the width direction."""
    if width <= 0:
        return 1.0
    scale = max(1.0, MIN_TEXT_WIDTH_PX / width)
    if width * scale > MAX_IMAGE_SIDE_PX:
        scale = MAX_IMAGE_SIDE_PX / width
    return scale


def _plan_segments(height: int, scale: float, row_std: np.ndarray | None) -> list[tuple[int, int]]:
    """Splits [0, height) into (top, bottom) source-pixel ranges, each
    ending up about TILE_HEIGHT_PX tall once scaled.

    Cuts are placed on rows that are locally blank (flat luminance across
    a band of three consecutive rows), searching near each even split so
    that a text line is never bisected and no line appears in two tiles.
    Falls back to the even split when no blank row is in range."""
    tile_h = max(1, int(TILE_HEIGHT_PX / scale))
    if height <= tile_h:
        return [(0, height)]

    candidates = np.array([], dtype=int)
    if row_std is not None and row_std.size == height and height > 2:
        threshold = max(3.0, 0.05 * float(np.percentile(row_std, 90)))
        blank = row_std <= threshold
        # cut c is valid when rows c-1, c and c+1 are all blank
        valid = blank[:-2] & blank[1:-1] & blank[2:]
        candidates = np.nonzero(valid)[0] + 1

    cuts: list[int] = []
    previous = 0
    for target in range(tile_h, height, tile_h):
        low = previous + MIN_SEGMENT_PX
        high = min(height - MIN_SEGMENT_PX, target + SPLIT_SEARCH_PX)
        if high <= low:
            break
        window = candidates[(candidates >= low) & (candidates <= high)]
        if window.size:
            cut = int(window[int(np.argmin(np.abs(window - target)))])
        elif row_std is not None:
            # Nothing genuinely blank in range: a gradient, a wallpaper
            # behind the page or a panel that spans the width makes every
            # row look busy. Settle for the flattest row available rather
            # than an arbitrary one, so a line of text still is not the cut.
            cut = int(low) + int(np.argmin(row_std[low:high]))
        else:
            cut = min(max(target, low), high)
        if cut <= previous:
            continue
        cuts.append(cut)
        previous = cut

    bounds = [0, *cuts, height]
    return list(zip(bounds[:-1], bounds[1:]))


def _normalize(im: Image.Image) -> Image.Image:
    """Drop the screenshot to a single channel, so what the model reads is
    the ink and never the color of the page behind it.

    Measured over white, black, sepia, blue, WhatsApp-green,
    terminal-green-on-black, amber-on-black, a gradient page, a dim dark
    theme, a washed-out low-contrast page, colored code tokens, colored
    links, chroma-only contrast (green on green, blue on blue) and text
    laid straight on a busy photo: raw color lost characters on four of
    them (84.6% to 97.6% word recall), grayscale read every one at 100%.

    Two tempting extras measured as no help, so they are deliberately not
    done: inverting a dark screenshot changed nothing (PaddleOCR already
    reads white-on-black as well as black-on-white), and stretching the
    levels to chase contrast cost a token on a gradient page while
    fixing nothing the plain conversion had not already fixed. So this
    stays a luminance conversion — no threshold, no knob to go wrong."""
    return im.convert("L")


def _row_std_for_split(im: Image.Image) -> np.ndarray:
    """Per-row luminance standard deviation on a horizontally-subsampled
    copy — cheap to compute, and enough to tell a blank band apart from a
    line of text. Expects the already-grayscale page.

    The subsample is level-stretched first, and only here: this signal
    decides where a tall screenshot is cut, so what matters is that a
    line of text stands out from the band around it. On a washed-out
    page a line is only a few gray levels off its background, the
    absolute floor in _plan_segments then calls it blank, and the cut
    lands in the middle of a word. Stretching costs nothing for the model
    — this array is never handed to it."""
    subsample = np.ascontiguousarray(np.asarray(im, dtype=np.uint8)[:, ::4])
    stretched = ImageOps.autocontrast(Image.fromarray(subsample), cutoff=1)
    return np.asarray(stretched, dtype=np.uint8).astype(np.float32).std(axis=1)


def _decode_tiles(image: ExtractedImage) -> list[np.ndarray]:
    """Decodes one screenshot into the tile(s) OCR should run on,
    normalized so the reading does not depend on the background color.

    Returns one array when the image already fits the tile budget, or
    several when it is tall. Peak memory stays at the source image plus a
    single tile — the grayscale page it crops from is a third the size of
    the RGB copy it replaces — so slicing does not weaken the OOM
    protection that MAX_IMAGE_SIDE_PX exists for."""
    try:
        with Image.open(io.BytesIO(image.data)) as im:
            page = _normalize(im)
            width, height = page.size
            if width <= 0 or height <= 0:
                raise PaddleOcrError("Image has no pixels.", code=OCR_CODE_INVALID_IMAGE)

            scale = _scale_for_width(width)
            row_std = _row_std_for_split(page) if height * scale > TILE_HEIGHT_PX else None
            segments = _plan_segments(height, scale, row_std)

            out_width = max(1, round(width * scale))
            tiles: list[np.ndarray] = []
            for top, bottom in segments:
                tile = page.crop((0, top, width, bottom))
                out_height = max(1, round((bottom - top) * scale))
                if tile.size != (out_width, out_height):
                    tile = tile.resize((out_width, out_height), Image.LANCZOS)
                tiles.append(np.array(tile.convert("RGB")))
            return tiles
    except PaddleOcrError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PaddleOcrError(f"Could not decode image for OCR: {exc}", code=OCR_CODE_INVALID_IMAGE) from exc


def _get_pool(settings: Settings) -> list[_PooledEngine]:
    """Builds the engine pool on first use (double-checked locking so
    concurrent callers don't each build their own pool), then reuses
    it for the life of the process."""
    global _engine_pool
    if _engine_pool is None:
        with _pool_lock:
            if _engine_pool is None:
                try:
                    size = max(1, settings.OCR_MAX_CONCURRENCY)
                    _engine_pool = [_PooledEngine(settings) for _ in range(size)]
                except Exception as exc:  # noqa: BLE001
                    raise PaddleOcrError(f"Failed to load PaddleOCR engine: {exc}", code=OCR_CODE_ENGINE_LOAD) from exc
    return _engine_pool


# One OCR semaphore for the whole process, not one per request. Keyed by
# event loop because asyncio primitives bind to the loop they are first
# awaited on, and the test suite runs a fresh loop per test.
_semaphores: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Semaphore]] = {}
_semaphores_lock = threading.Lock()


def _get_semaphore(settings: Settings) -> asyncio.Semaphore:
    """Returns the process-wide OCR semaphore for the running loop.

    run_ocr_on_images used to build a new Semaphore on every call, so the
    bound it implements held only *within* one request. Under Fluid
    Compute several requests share a single instance and event loop, and
    each arrived with its own permits: N screenshots decoding at once
    while only the engine lock serialized predict — every waiting request
    parked its full tile set in memory, which is exactly the peak the
    semaphore exists to prevent on a 2 GB host. Sharing one restores the
    documented bound across concurrent requests."""
    loop = asyncio.get_running_loop()
    key = id(loop)
    with _semaphores_lock:
        entry = _semaphores.get(key)
        if entry is not None and entry[0].is_closed():
            # Loop is gone (test teardown); don't accumulate entries.
            del _semaphores[key]
            entry = None
        if entry is None:
            entry = (loop, asyncio.Semaphore(max(1, settings.OCR_MAX_CONCURRENCY)))
            _semaphores[key] = entry
        return entry[1]


def _sort_key_from_box(box: Any) -> tuple[float, float]:
    """Returns a (y, x) reading-order sort key from either an
    [x1, y1, x2, y2] axis-aligned box or a 4-point polygon
    [[x, y], [x, y], [x, y], [x, y]]."""
    first = box[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        return (min(ys), min(xs))
    x1, y1, _x2, _y2 = box
    return (float(y1), float(x1))


def _box_bounds(box: Any) -> tuple[float, float, float, float]:
    """Normalizes an [x1, y1, x2, y2] box or a 4-point polygon into
    (x1, y1, x2, y2) pixel bounds."""
    first = box[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        return min(xs), min(ys), max(xs), max(ys)
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return float(x1), float(y1), float(x2), float(y2)


# Reading-order tuning. A "box" is one recognized fragment; several
# fragments routinely belong to one visual line (code is split at
# indentation), and a split-screen screenshot holds two text columns.
_COLUMN_GUTTER_PX = 60.0
_MIN_COLUMN_BOXES = 3
_COLUMN_Y_OVERLAP = 0.25
_MAX_COLUMN_DEPTH = 2
# A fixed icon rail hugs the left edge of a browser screenshot with only
# ~25px of clearance — too tight for the gutter above, yet wide enough
# that its labels (`Notes`, `Q 46`) otherwise weld themselves onto the
# first sentence of the description. Split it off when it really is a
# narrow strip: few boxes, all hugging the edge.
_NARROW_GUTTER_PX = 24.0
_NARROW_WIDTH_PX = 80.0
_NARROW_MAX_BOXES = 8
# A real text column carries several boxes, so a stray box is not worth
# splitting off. An icon rail is shorter than that: with a near-black
# background the detector reads its labels but skips its glyphs, leaving
# as few as two, and without this exception both get welded onto the
# first sentence of the statement instead of standing above it.
_MIN_NARROW_COLUMN_BOXES = 2
_LINE_OVERLAP_RATIO = 0.5
_WORD_GAP_RATIO = 0.15
_MIN_GAP_PX = 4.0


def _split_columns(
    items: list[tuple[str, float, float, float, float]], depth: int = 0
) -> list[list[tuple[str, float, float, float, float]]]:
    """Splits recognized fragments into left-to-right text columns.

    Screenshots of split-view pages (problem on the left, code editor on
    the right) produce two independent columns of text. A naive top-to-
    bottom sort interleaves them line by line and hands the model
    nonsense, so we look for a vertical gutter no box crosses and read
    each side on its own.
    """
    if depth >= _MAX_COLUMN_DEPTH or len(items) < 2 * _MIN_COLUMN_BOXES:
        return [items]

    split_at: float | None = None
    best_gap = 0.0
    ordered = sorted(items, key=lambda it: it[1])
    group_min_x1 = ordered[0][1]
    group_count = 0
    running_max_x2 = -1.0
    for _text, x1, _y1, x2, _y2 in ordered:
        if running_max_x2 >= 0.0:
            gap = x1 - running_max_x2
            narrow = (
                gap >= _NARROW_GUTTER_PX
                and running_max_x2 - group_min_x1 <= _NARROW_WIDTH_PX
                and group_count <= _NARROW_MAX_BOXES
            )
            if (gap >= _COLUMN_GUTTER_PX or narrow) and gap > best_gap:
                best_gap = gap
                split_at = running_max_x2
        group_count += 1
        running_max_x2 = max(running_max_x2, x2)

    if split_at is None:
        return [items]

    left = [it for it in items if it[3] <= split_at]
    right = [it for it in items if it[1] > split_at]
    if len(right) < _MIN_COLUMN_BOXES:
        return [items]
    narrow_strip = bool(left) and max(it[3] for it in left) <= _NARROW_WIDTH_PX
    if len(left) < _MIN_COLUMN_BOXES and not (
        narrow_strip and len(left) >= _MIN_NARROW_COLUMN_BOXES
    ):
        return [items]

    def y_span(group: list[tuple[str, float, float, float, float]]) -> float:
        return max(it[4] for it in group) - min(it[2] for it in group)

    overlap = min(max(it[4] for it in left), max(it[4] for it in right)) - max(
        min(it[2] for it in left), min(it[2] for it in right)
    )
    if overlap < _COLUMN_Y_OVERLAP * max(y_span(left), y_span(right)):
        # Vertically disjoint groups are just one column with a wide
        # margin — reading them left-to-right would reorder the text.
        return [items]

    return _split_columns(left, depth + 1) + _split_columns(right, depth + 1)


def _group_into_lines(
    items: list[tuple[str, float, float, float, float]],
) -> list[list[tuple[str, float, float, float, float]]]:
    """Buckets fragments into visual lines by vertical overlap, then
    orders each line left-to-right.

    Sorting fragments straight by their top edge scrambles any line whose
    glyphs differ in height: `return []` came back as `[]` + `return`
    because brackets start above the lowercase letters."""
    lines: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda it: (it[2] + it[4]) / 2.0):
        box_h = item[4] - item[2]
        best: dict[str, Any] | None = None
        best_overlap = 0.0
        for line in lines:
            line_h = line["y2"] - line["y1"]
            overlap = min(line["y2"], item[4]) - max(line["y1"], item[2])
            if overlap >= _LINE_OVERLAP_RATIO * min(box_h, line_h) and overlap > best_overlap:
                best = line
                best_overlap = overlap
        if best is None:
            lines.append({"y1": item[2], "y2": item[4], "items": [item]})
        else:
            best["items"].append(item)
            best["y1"] = min(best["y1"], item[2])
            best["y2"] = max(best["y2"], item[4])

    lines.sort(key=lambda line: (line["y1"] + line["y2"]) / 2.0)
    ordered: list[list[tuple[str, float, float, float, float]]] = []
    for line in lines:
        ordered.append(sorted(line["items"], key=lambda it: it[1]))
    return ordered


def _join_line(
    items: list[tuple[str, float, float, float, float]],
) -> str:
    """Concatenates one visual line's fragments, inserting a space only
    where the horizontal gap looks like a real word space rather than a
    break det made inside a token."""
    parts: list[str] = []
    prev_x2: float | None = None
    prev_h = _MIN_GAP_PX
    for text, x1, y1, x2, y2 in items:
        height = max(_MIN_GAP_PX, y2 - y1)
        if prev_x2 is None:
            parts.append(text)
        else:
            gap = x1 - prev_x2
            threshold = _WORD_GAP_RATIO * min(prev_h, height)
            parts.append((" " if gap > threshold else "") + text)
        prev_x2 = x2
        prev_h = height
    return "".join(parts).strip()


def _lines_in_reading_order(result: Any) -> list[str]:
    """Pulls recognized text out of one PaddleOCR `.predict()` result and
    orders it the way a person reads a screenshot: column by column
    (split-view pages), line by line (top-to-bottom), and left-to-right
    within each line."""
    res = result[0] if isinstance(result, (list, tuple)) else result
    texts = list(res.get("rec_texts", []) or [])
    if not texts:
        return []

    boxes = res.get("rec_boxes", None)
    if boxes is None:
        boxes = res.get("rec_polys", None) or res.get("dt_polys", None)

    if boxes is None or len(boxes) != len(texts):
        # No usable geometry to sort by — fall back to whatever order
        # PaddleOCR returned rather than dropping the text.
        return [t for t in texts if t and t.strip()]

    items = [
        (text.strip(), *_box_bounds(box))
        for text, box in zip(texts, boxes)
        if text and text.strip()
    ]
    if not items:
        return []

    lines: list[str] = []
    for column in _split_columns(items):
        for line in _group_into_lines(column):
            joined = _join_line(line)
            if joined:
                lines.append(joined)
    return lines


def _run_predict(engine: _PooledEngine, image_array: np.ndarray) -> list[str]:
    with engine.lock:
        try:
            result = engine.engine.predict(image_array)
        except Exception as exc:  # noqa: BLE001
            raise PaddleOcrError(f"PaddleOCR inference failed: {exc}", code=OCR_CODE_INFERENCE) from exc
    return _lines_in_reading_order(result)


async def _ocr_single_image(
    index: int,
    image: ExtractedImage,
    pool: list[_PooledEngine],
    semaphore: asyncio.Semaphore,
) -> OcrResult:
    # Decode *inside* the semaphore, not before it: with N images the
    # unbounded version held N full-resolution RGB arrays at once while
    # only OCR_MAX_CONCURRENCY of them were actually being processed.
    # Peak memory is now bounded by the semaphore, not by image count —
    # and by MAX_IMAGE_SIDE_PX within each screenshot, since a tall image
    # is processed one tile at a time and each tile is released as soon
    # as it has been read.
    async with semaphore:
        tiles = await asyncio.to_thread(_decode_tiles, image)
        engine = pool[index % len(pool)]
        lines: list[str] = []
        try:
            for position, tile in enumerate(tiles):
                lines.extend(await asyncio.to_thread(_run_predict, engine, tile))
                tiles[position] = None  # type: ignore[call-overload]
        finally:
            tiles.clear()

    text = "\n".join(lines).strip()
    if not text:
        return OcrResult(index=index, text="", unreadable=True)
    return OcrResult(index=index, text=text, unreadable=False)


async def run_ocr_on_images(
    settings: Settings,
    images: list[ExtractedImage],
    client: object | None = None,  # unused; kept for signature compatibility with call sites
) -> list[OcrResult]:
    """Runs OCR on all images concurrently (bounded by the local
    engine pool size), preserving input order in the returned list
    regardless of completion order."""
    if not images:
        return []

    # Off the event loop: building the pool loads models and can hold the
    # engine lock for tens of seconds while the startup warm-up is still
    # running, and a request that waits on it *synchronously* freezes
    # every other request in the instance.
    pool = await asyncio.to_thread(_get_pool, settings)
    semaphore = _get_semaphore(settings)
    tasks = [_ocr_single_image(idx, img, pool, semaphore) for idx, img in enumerate(images)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r.index)


async def warm_up(settings: Settings) -> None:
    """Eagerly loads (and caches) the PaddleOCR engine pool, so the
    first real request doesn't pay model load/download latency.
    Intended to be called once from the app startup event."""
    await asyncio.to_thread(_get_pool, settings)
