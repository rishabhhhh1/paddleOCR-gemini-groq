import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.models.solver import ExtractedImage
from app.services.paddle_ocr import (
    MAX_IMAGE_SIDE_PX,
    MIN_TEXT_WIDTH_PX,
    TILE_HEIGHT_PX,
    _decode_tiles,
    _normalize,
    _plan_segments,
    _row_std_for_split,
    _scale_for_width,
    _ocr_single_image,
)

_SANS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in _SANS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _image_bytes(width: int, height: int, color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _extracted(width: int, height: int) -> ExtractedImage:
    return ExtractedImage(mime_type="image/png", data=_image_bytes(width, height))


def test_scale_upscales_narrow_screenshots():
    assert _scale_for_width(480) == MIN_TEXT_WIDTH_PX / 480
    assert _scale_for_width(600) == MIN_TEXT_WIDTH_PX / 600


def test_scale_leaves_typical_screenshots_untouched():
    assert _scale_for_width(1170) == 1.0
    assert _scale_for_width(1920) == 1.0


def test_scale_never_exceeds_the_width_cap():
    assert _scale_for_width(4000) == MAX_IMAGE_SIDE_PX / 4000
    assert 4000 * _scale_for_width(4000) == MAX_IMAGE_SIDE_PX


def test_short_image_is_a_single_tile():
    assert _plan_segments(900, 1.0, None) == [(0, 900)]


def test_tall_image_is_sliced_into_tile_sized_segments():
    height = int(TILE_HEIGHT_PX * 3.4)
    segments = _plan_segments(height, 1.0, None)
    assert len(segments) == 4
    assert segments[0][0] == 0
    assert segments[-1][1] == height
    # contiguous, no gaps or overlaps
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
    assert all(bottom - top >= 400 for top, bottom in segments)


def test_cuts_land_on_blank_rows_not_through_text():
    # 6000 rows: text bands of 40 rows, blank gaps of 20 rows
    row_std = np.array([50.0 if (i % 60) < 40 else 0.0 for i in range(6000)])
    segments = _plan_segments(6000, 1.0, row_std)
    for _top, bottom in segments[:-1]:
        # the row just above every cut must be blank
        assert row_std[bottom - 1] <= 3.0, f"cut at {bottom} lands on text"


def test_faint_text_still_counts_as_ink():
    """Where a tall page is cut is decided by per-row variance against a
    fixed blank threshold. Text that is only a few gray levels off its
    background scores under that threshold raw, which would read the
    line as a blank band and cut straight through it."""
    img = Image.new("RGB", (900, 6000), "#ffffff")
    draw = ImageDraw.Draw(img)
    font = _font(18)
    y = 20
    while y < 5980:
        draw.text((30, y), "The quick brown fox jumps over the lazy dog again",
                  fill="#f6f6f6", font=font)
        y += 60

    row_std = _row_std_for_split(_normalize(img))
    # every row through the middle of a text band carries ink ...
    assert row_std[24:37].min() > 3.0
    # ... and the band above it is genuinely blank.
    assert row_std[0:13].max() == 0.0
    for _top, bottom in _plan_segments(6000, 1.0, row_std):
        assert row_std[bottom - 1] <= 3.0, f"cut at {bottom} lands on text"


def test_narrow_image_is_upscaled_before_ocr():
    tiles = _decode_tiles(_extracted(480, 400))
    assert len(tiles) == 1
    assert tiles[0].shape[1] >= MIN_TEXT_WIDTH_PX


def test_tall_image_is_sliced_and_every_tile_is_bounded():
    tiles = _decode_tiles(_extracted(1170, int(TILE_HEIGHT_PX * 2.5)))
    assert len(tiles) >= 3
    for tile in tiles:
        longest = max(tile.shape[0], tile.shape[1])
        assert longest <= MAX_IMAGE_SIDE_PX
        # text stays at native size: the width is not shrunk
        assert tile.shape[1] == 1170


def test_corrupt_image_raises_a_domain_error():
    from app.services.paddle_ocr import OCR_CODE_INVALID_IMAGE, PaddleOcrError

    bad = ExtractedImage(mime_type="image/png", data=b"not-an-image")
    try:
        _decode_tiles(bad)
    except PaddleOcrError as exc:
        # Tagged as a client-input problem so the routes answer 400
        # instead of blaming an upstream OCR provider with a 502.
        assert exc.code == OCR_CODE_INVALID_IMAGE
    else:
        raise AssertionError("expected PaddleOcrError")


def test_truncated_jpeg_is_tagged_as_a_client_error():
    """verify() does not fully decode, so a truncated JPEG clears image
    validation and only dies here — it must arrive tagged as the
    client's problem, not as an upstream failure."""
    import io

    from PIL import Image, ImageDraw

    from app.services.paddle_ocr import OCR_CODE_INVALID_IMAGE, PaddleOcrError

    img = Image.new("RGB", (640, 480), (255, 255, 255))
    ImageDraw.Draw(img).text((20, 20), "problem statement", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    truncated = buf.getvalue()[: int(len(buf.getvalue()) * 0.4)]

    try:
        _decode_tiles(ExtractedImage(mime_type="image/jpeg", data=truncated))
    except PaddleOcrError as exc:
        assert exc.code == OCR_CODE_INVALID_IMAGE
        assert "truncated" in str(exc)
    else:
        raise AssertionError("expected PaddleOcrError for a truncated JPEG")


@pytest.mark.asyncio
async def test_ocr_single_image_concatenates_tiles_in_order(monkeypatch):
    from app.services.paddle_ocr import _PooledEngine

    tiles = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr("app.services.paddle_ocr._decode_tiles", lambda img: list(tiles))

    calls = {"n": 0}

    def fake_predict(engine, arr):
        calls["n"] += 1
        return [f"line from tile {calls['n']}"]

    monkeypatch.setattr("app.services.paddle_ocr._run_predict", fake_predict)

    class FakeEngine:
        pass

    result = await _ocr_single_image(
        0,
        _extracted(10, 10),
        [FakeEngine()],  # type: ignore[list-item]
        __import__("asyncio").Semaphore(1),
    )
    assert calls["n"] == 3
    assert result.text == "line from tile 1\nline from tile 2\nline from tile 3"
    assert not result.unreadable


def _rail_and_body(rail: list[tuple[str, float, float, float, float]]):
    body = [
        (f"statement line {i}", 95.0, 30.0 + i * 110, 700.0, 60.0 + i * 110)
        for i in range(6)
    ]
    return rail + body


def test_a_short_icon_rail_is_still_its_own_column():
    """The detector skips the glyphs on a near-black rail and returns as
    little as two labels for it — two is still a column, and anything
    fewer leaves both welded onto the first sentence of the statement."""
    from app.services.paddle_ocr import _split_columns

    rail = [
        ("Q 46", 13.0, 125.0, 50.0, 144.0),
        ("Notes", 10.0, 696.0, 50.0, 717.0),
    ]
    columns = _split_columns(_rail_and_body(rail))
    assert len(columns) == 2
    assert [item[0] for item in columns[0]] == ["Q 46", "Notes"]
    assert columns[1][0][0] == "statement line 0"


def test_a_lone_box_at_the_left_edge_is_not_a_column():
    from app.services.paddle_ocr import _split_columns

    columns = _split_columns(_rail_and_body([("Notes", 10.0, 696.0, 50.0, 717.0)]))
    assert len(columns) == 1


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_ocr_bound(monkeypatch):
    """The bound has to hold across requests, not just within one.

    Under Fluid Compute several requests share an instance, so a fresh
    semaphore per request let them all decode at once — the peak-memory
    condition the semaphore exists to prevent."""
    import asyncio
    import threading
    import time

    from app.config import get_settings
    from app.services import paddle_ocr as po

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def slow_decode(img):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return [np.zeros((4, 4, 3), dtype=np.uint8)]

    class FakeEngine:
        pass

    monkeypatch.setattr("app.services.paddle_ocr._decode_tiles", slow_decode)
    monkeypatch.setattr("app.services.paddle_ocr._run_predict", lambda engine, arr: ["line"])
    monkeypatch.setattr("app.services.paddle_ocr._get_pool", lambda settings: [FakeEngine()])

    settings = get_settings()
    await asyncio.gather(
        po.run_ocr_on_images(settings, [_extracted(10, 10)]),
        po.run_ocr_on_images(settings, [_extracted(10, 10)]),
        po.run_ocr_on_images(settings, [_extracted(10, 10)]),
    )
    assert state["peak"] == 1, f"OCR bound leaked across requests: peak={state['peak']}"
