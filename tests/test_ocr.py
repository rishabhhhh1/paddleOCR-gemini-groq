from tests.conftest import make_test_image_data_url


def _mock_ocr(monkeypatch):
    import app.api.ocr as ocr_module
    from app.models.solver import OcrResult

    async def fake_ocr(settings, images, client=None):
        return [OcrResult(index=i, text=f"line from image {i}") for i in range(len(images))]

    monkeypatch.setattr(ocr_module, "run_ocr_on_images", fake_ocr)


def _batch(images, start_index=0):
    return {"images": images, "start_index": start_index}


def test_ocr_batch_returns_globally_numbered_results(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    img = make_test_image_data_url("PNG")

    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch([img, img]),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "ocr.result"
    assert body["start_index"] == 0
    assert body["count"] == 2
    assert body["next_start_index"] == 2
    assert [r["index"] for r in body["results"]] == [0, 1]
    assert "SCREENSHOT 1" in body["text"]
    assert "SCREENSHOT 2" in body["text"]


def test_ocr_batches_keep_numbering_across_calls(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    img = make_test_image_data_url("PNG")

    first = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch([img, img]),
    ).json()

    second = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch([img], start_index=first["next_start_index"]),
    ).json()

    assert [r["index"] for r in second["results"]] == [2]
    assert second["text"].startswith("SCREENSHOT 3")
    assert "SCREENSHOT 1" not in second["text"]

    combined = first["text"] + "\n\n" + second["text"]
    assert "SCREENSHOT 1" in combined and "SCREENSHOT 3" in combined


def test_ocr_rejects_more_images_than_batch_limit(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    img = make_test_image_data_url("PNG")

    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch([img] * 5),
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["message"].startswith("Request contains 5 images")


def test_ocr_requires_api_key(app_client, monkeypatch):
    _mock_ocr(monkeypatch)
    resp = app_client.post("/v1/ocr", json=_batch([make_test_image_data_url("PNG")]))
    assert resp.status_code == 401


def test_ocr_rejects_bad_image_data(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch(["data:image/png;base64,not-valid-base64!!!"]),
    )
    assert resp.status_code == 400


def _large_image_data_url(mb_target=2.0, seed=0):
    """A noisy JPEG big enough that two of them blow MAX_OCR_BATCH_BYTES."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_ocr_batch_over_byte_budget_rejected(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    big = _large_image_data_url()
    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch([big, big]),
    )
    assert resp.status_code == 413
    err = resp.json()["error"]
    assert err["code"] == "request_too_large"
    assert "Split it into smaller batches" in err["message"]


def test_ocr_batch_stitch_false_uses_unstitched_text(app_client, backend_api_key, monkeypatch):
    """stitch=False must reach format_screenshot_blocks, not just be accepted
    and ignored -- verified indirectly: two OCR results whose (mocked) text
    would be marked DUPLICATE under stitching must not be, once stitch=False
    is threaded through to the response's `text` field."""
    import app.api.ocr as ocr_module
    from app.models.solver import OcrResult

    async def fake_ocr(settings, images, client=None):
        return [OcrResult(index=i, text="Repeated content line one\nline two") for i in range(len(images))]

    monkeypatch.setattr(ocr_module, "run_ocr_on_images", fake_ocr)
    img = make_test_image_data_url("PNG")

    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"images": [img, img], "start_index": 0, "stitch": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "DUPLICATE" not in body["text"]
    assert body["text"].count("Repeated content line one") == 2


def test_unreadable_screenshot_is_a_400_not_a_502(app_client, backend_api_key, monkeypatch):
    """A screenshot the server cannot decode is a bad request. Validation
    misses it because verify() never fully decodes, so it surfaces from
    inside run_ocr_on_images — it must not be reported as an upstream OCR
    outage."""
    import app.api.ocr as ocr_module
    from app.services.paddle_ocr import OCR_CODE_INVALID_IMAGE, PaddleOcrError

    async def failing_ocr(settings, images, client=None):
        raise PaddleOcrError(
            "Could not decode image for OCR: image file is truncated (28 bytes not processed)",
            code=OCR_CODE_INVALID_IMAGE,
        )

    monkeypatch.setattr(ocr_module, "run_ocr_on_images", failing_ocr)

    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch([make_test_image_data_url("PNG")]),
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_image"
    assert "truncated" in err["message"]


def test_ocr_upstream_failure_carries_a_diagnostic_code(app_client, backend_api_key, monkeypatch):
    """The 502 body is deliberately generic, so error.code is what tells a
    client (and us, from a screenshot of the response) which stage broke."""
    import app.api.ocr as ocr_module
    from app.services.paddle_ocr import OCR_CODE_INFERENCE, PaddleOcrError

    async def failing_ocr(settings, images, client=None):
        raise PaddleOcrError("PaddleOCR inference failed: boom", code=OCR_CODE_INFERENCE)

    monkeypatch.setattr(ocr_module, "run_ocr_on_images", failing_ocr)

    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json=_batch([make_test_image_data_url("PNG")]),
    )
    assert resp.status_code == 502
    err = resp.json()["error"]
    assert err["type"] == "upstream_error"
    assert err["code"] == "ocr_inference_failed"
