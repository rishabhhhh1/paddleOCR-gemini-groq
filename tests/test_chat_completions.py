from tests.conftest import make_test_image_data_url

FAKE_SOLUTION = (
    "## Problem\n...\n## Key Observation\n...\n## Approach\n...\n## Algorithm\n1. ...\n"
    "## Correctness\n...\n## Complexity\nTime: O(n)\nSpace: O(1)\n## Code\n```cpp\nint main(){}\n```\n"
    "## Edge Cases\n...\n## Warnings\n..."
)


def _mock_solver(monkeypatch, captured=None):
    import app.api.chat_completions as cc

    async def fake_solve(settings, text, client=None, max_tokens=None):
        if captured is not None:
            captured.append(text)
        return FAKE_SOLUTION

    monkeypatch.setattr(cc, "solve_problem", fake_solve)


def _mock_ocr(monkeypatch):
    import app.api.chat_completions as cc
    from app.models.solver import OcrResult

    async def fake_ocr(settings, images, client=None):
        return [OcrResult(index=i, text=f"extracted text {i}") for i in range(len(images))]

    monkeypatch.setattr(cc, "run_ocr_on_images", fake_ocr)


def test_client_max_tokens_is_forwarded_to_the_solver(app_client, backend_api_key, monkeypatch):
    captured = {}
    import app.api.chat_completions as cc

    async def fake_solve(settings, text, client=None, max_tokens=None):
        captured["max_tokens"] = max_tokens
        return FAKE_SOLUTION

    monkeypatch.setattr(cc, "solve_problem", fake_solve)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Solve: two sum"}],
        },
    )
    assert resp.status_code == 200
    assert captured["max_tokens"] == 2048


def test_absent_max_tokens_falls_back_to_the_server_budget(app_client, backend_api_key, monkeypatch):
    captured = {"sentinel": object()}
    import app.api.chat_completions as cc

    async def fake_solve(settings, text, client=None, max_tokens=None):
        captured["max_tokens"] = max_tokens
        return FAKE_SOLUTION

    monkeypatch.setattr(cc, "solve_problem", fake_solve)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"model": "dsa-solver", "messages": [{"role": "user", "content": "Solve: two sum"}]},
    )
    assert resp.status_code == 200
    assert captured["max_tokens"] is None


def test_truncated_answer_surfaces_as_502_and_is_logged(app_client, backend_api_key, monkeypatch, caplog):
    import logging

    import app.api.chat_completions as cc
    from app.services.consensus_solver import ConsensusSolverError

    async def failing_solve(settings, text, client=None, max_tokens=None):
        raise ConsensusSolverError(
            "Both solver providers failed. "
            "Gemini: Gemini stopped at its 32768-token output cap before finishing "
            "the answer. Grok: Grok stopped at its 32768-token output cap."
        )

    monkeypatch.setattr(cc, "solve_problem", failing_solve)

    with caplog.at_level(logging.WARNING, logger="dsa_practice_solver"):
        resp = app_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {backend_api_key}"},
            json={"model": "dsa-solver", "messages": [{"role": "user", "content": "Solve"}]},
        )

    assert resp.status_code == 502
    # The reason must never reach the client...
    assert "32768" not in resp.text
    # ...but it must reach the logs, or a truncated answer is undebuggable.
    assert any("output cap" in record.getMessage() for record in caplog.records)


def test_text_only_request_does_not_invoke_ocr(app_client, backend_api_key, monkeypatch):
    captured = []
    _mock_solver(monkeypatch, captured)

    import app.api.chat_completions as cc

    async def fail_if_called(*a, **kw):
        raise AssertionError("OCR must not be invoked for text-only requests")

    monkeypatch.setattr(cc, "run_ocr_on_images", fail_if_called)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"model": "dsa-solver", "messages": [{"role": "user", "content": "Solve: two sum, n <= 1e5"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "dsa-solver"
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == FAKE_SOLUTION
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "Solve: two sum" in captured[0]


def test_multimodal_request_invokes_ocr_then_solver(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    captured = []
    _mock_solver(monkeypatch, captured)

    img = make_test_image_data_url("PNG")
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Solve this in C++17."},
                        {"type": "image_url", "image_url": {"url": img}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert "SCREENSHOT 1" in captured[0]
    assert "ADDITIONAL USER NOTES" in captured[0]


def test_multiple_images_preserve_order(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    captured = []
    _mock_solver(monkeypatch, captured)

    imgs = [make_test_image_data_url("PNG") for _ in range(3)]
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": u}} for u in imgs],
                }
            ],
        },
    )
    assert resp.status_code == 200
    text = captured[0]
    assert text.index("SCREENSHOT 1") < text.index("SCREENSHOT 2") < text.index("SCREENSHOT 3")


def test_wrong_model_name_rejected(app_client, backend_api_key, monkeypatch):
    _mock_solver(monkeypatch)
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "model_not_found"


def test_streaming_returns_unsupported_error(app_client, backend_api_key, monkeypatch):
    _mock_solver(monkeypatch)
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"model": "dsa-solver", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_feature"


def test_mocked_solver_failure_returns_502(app_client, backend_api_key, monkeypatch):
    import app.api.chat_completions as cc
    from app.services.consensus_solver import ConsensusSolverError

    async def failing_solve(settings, text, client=None, max_tokens=None):
        raise ConsensusSolverError("boom")

    monkeypatch.setattr(cc, "solve_problem", failing_solve)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"model": "dsa-solver", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_error"
    assert resp.json()["error"]["code"] == "solver_failed"


def test_unreadable_screenshot_is_a_400_not_a_502(app_client, backend_api_key, monkeypatch):
    import app.api.chat_completions as cc
    from app.services.paddle_ocr import OCR_CODE_INVALID_IMAGE, PaddleOcrError

    async def failing_ocr(settings, images, client=None):
        raise PaddleOcrError("Could not decode image for OCR: image file is truncated", code=OCR_CODE_INVALID_IMAGE)

    monkeypatch.setattr(cc, "run_ocr_on_images", failing_ocr)
    _mock_solver(monkeypatch)

    img = make_test_image_data_url("PNG")
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}]}],
        },
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_image"


def test_mocked_ocr_failure_returns_502(app_client, backend_api_key, monkeypatch):
    import app.api.chat_completions as cc
    from app.services.paddle_ocr import PaddleOcrError

    async def failing_ocr(settings, images, client=None):
        raise PaddleOcrError("boom")

    monkeypatch.setattr(cc, "run_ocr_on_images", failing_ocr)
    _mock_solver(monkeypatch)

    img = make_test_image_data_url("PNG")
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}]}],
        },
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "ocr_inference_failed"


def test_solve_from_previously_ocrd_results_skips_ocr(app_client, backend_api_key, monkeypatch):
    captured = []
    _mock_solver(monkeypatch, captured)

    import app.api.chat_completions as cc

    async def fail_if_called(*a, **kw):
        raise AssertionError("OCR must not re-run on results already returned by /v1/ocr")

    monkeypatch.setattr(cc, "run_ocr_on_images", fail_if_called)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": "Solve this in C++17."}],
            "ocr_results": [
                {"index": 0, "text": "first screenshot text"},
                {"index": 1, "text": "second screenshot text"},
            ],
        },
    )
    assert resp.status_code == 200
    text = captured[0]
    assert "2 screenshot(s)" in text
    assert text.index("SCREENSHOT 1") < text.index("SCREENSHOT 2")
    assert "first screenshot text" in text and "second screenshot text" in text
    assert "ADDITIONAL USER NOTES" in text


def test_ocr_results_merge_with_inline_images_in_upload_order(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    captured = []
    _mock_solver(monkeypatch, captured)

    img = make_test_image_data_url("PNG")
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}]}],
            # Prior batches carry global indices; fresh OCR restarts at 0.
            "ocr_results": [
                {"index": 5, "text": "batch one"},
                {"index": 6, "text": "batch two"},
            ],
        },
    )
    assert resp.status_code == 200
    text = captured[0]
    assert "3 screenshot(s)" in text
    assert "SCREENSHOT 1\nbatch one" in text
    assert "SCREENSHOT 2\nbatch two" in text
    assert "SCREENSHOT 3\nextracted text 0" in text


def test_too_many_screenshots_across_batches_rejected(app_client, backend_api_key, monkeypatch):
    captured = []
    _mock_solver(monkeypatch, captured)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": "Solve"}],
            "ocr_results": [{"index": i, "text": f"s {i}"} for i in range(21)],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "too_many_images"
    assert not captured


def test_prompt_over_char_limit_rejected(app_client, backend_api_key, monkeypatch):
    captured = []
    _mock_solver(monkeypatch, captured)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": "Solve"}],
            "ocr_results": [{"index": 0, "text": "x" * 300_000}],
        },
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "prompt_too_large"
    assert not captured


def test_inline_images_over_byte_budget_point_to_two_phase_flow(app_client, backend_api_key, monkeypatch):
    import base64
    import io

    import numpy as np
    from PIL import Image

    _mock_ocr(monkeypatch)
    _mock_solver(monkeypatch)

    def big(seed):
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": big(1)}}, {"type": "image_url", "image_url": {"url": big(2)}}]}
            ],
        },
    )
    assert resp.status_code == 413
    err = resp.json()["error"]
    assert err["code"] == "request_too_large"
    assert "/v1/ocr" in err["message"]


def test_stitch_false_prevents_cross_screenshot_dedup(app_client, backend_api_key, monkeypatch):
    """Two images that OCR to the same boilerplate-looking text: with the
    default stitching this collapses into a DUPLICATE marker; stitch=False
    must pass both through untouched to the solver."""
    import app.api.chat_completions as cc
    from app.models.solver import OcrResult

    async def fake_ocr(settings, images, client=None):
        return [OcrResult(index=i, text="Constraints:\n1 <= n <= 1e5\nSame line") for i in range(len(images))]

    monkeypatch.setattr(cc, "run_ocr_on_images", fake_ocr)

    captured = []
    _mock_solver(monkeypatch, captured)

    img = make_test_image_data_url("PNG")
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img}},
                        {"type": "image_url", "image_url": {"url": img}},
                    ],
                }
            ],
            "stitch": False,
        },
    )
    assert resp.status_code == 200
    solved_text = captured[0]
    assert "DUPLICATE" not in solved_text
    assert solved_text.count("Same line") == 2
