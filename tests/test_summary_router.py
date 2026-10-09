"""summary router extraction (#622): routes live in routers/summary.py, API surface unchanged."""

from academic_paper.routers.summary import router
from academic_paper.server import app

EXPECTED = [
    ("GET", "/papers/{paper_id}/summary"),
    ("POST", "/papers/{paper_id}/summary"),
    ("POST", "/papers/score-all"),
    ("POST", "/papers/{paper_id}/score"),
    ("GET", "/summaries"),
]


def test_summary_routes_registered_in_order_and_in_openapi():
    paths = app.openapi()["paths"]
    for method, path in EXPECTED:
        assert method.lower() in paths[path]
    assert [(next(iter(r.methods)), r.path) for r in router.routes] == EXPECTED


def test_route_resolution_order_score_all_before_paper_id_routes():
    """/papers/score-all stays declared before the /papers/{paper_id}/... routes (OpenAPI keeps declaration order)."""
    order = list(app.openapi()["paths"])
    assert order.index("/papers/score-all") < order.index("/papers/{paper_id}/score")
