"""papers router extraction (#617): routes live in routers/papers.py, API surface unchanged."""

from academic_paper.routers.papers import router
from academic_paper.server import app


def test_papers_routes_registered_on_router_and_in_openapi():
    paths = app.openapi()["paths"]
    assert "post" in paths["/papers/ingest"]
    assert "get" in paths["/papers"]
    assert "get" in paths["/papers/{paper_id}"]
    assert {r.path for r in router.routes} == {"/papers/ingest", "/papers", "/papers/{paper_id}"}

