def test_health(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # The live build's short SHA (or "unknown" outside Vercel) — this is
    # how a response shape change gets confirmed to be deployed.
    assert body["commit"]


def test_health_head(app_client):
    # Uptime pingers (e.g. UptimeRobot free tier) send HEAD requests and
    # can't be reconfigured to use GET, so /health must accept HEAD too.
    resp = app_client.head("/health")
    assert resp.status_code == 200
