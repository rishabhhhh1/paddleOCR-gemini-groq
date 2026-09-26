import os

from fastapi import APIRouter

router = APIRouter(tags=["health"])


# Explicitly declare both GET and HEAD. FastAPI/Starlette does NOT
# automatically add HEAD support to a GET-only route, so uptime pingers
# that send HEAD requests (e.g. UptimeRobot on its free tier, which
# doesn't allow choosing GET) get a 405 Method Not Allowed unless HEAD
# is registered here too.
@router.get("/health")
@router.head("/health")
async def health():
    # Which build is actually serving. An error contract change is
    # invisible otherwise — a client still reporting the old response
    # shape after a deploy looks exactly like a deploy that never
    # happened, and there was no way to tell those apart from outside.
    sha = os.environ.get("VERCEL_GIT_COMMIT_SHA", "")
    return {"status": "ok", "commit": sha[:12] if sha else "unknown"}
