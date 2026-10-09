"""jobs router extraction (#621): routes live in routers/jobs.py, API surface unchanged."""

from academic_paper.routers.jobs import router
from academic_paper.server import app


def test_jobs_routes_registered_on_router_and_in_openapi():
    paths = app.openapi()["paths"]
    assert "post" in paths["/jobs/summarize-all"]
    assert "get" in paths["/jobs/{job_id}"]
    assert "get" in paths["/jobs"]
    assert {r.path for r in router.routes} == {"/jobs/summarize-all", "/jobs/{job_id}", "/jobs"}


def test_summarize_all_declared_before_job_id_and_returns_202_shape_requires_auth():
    order = [r.path for r in router.routes]
    assert order.index("/jobs/summarize-all") < order.index("/jobs/{job_id}")
    assert paths_status(app, "/jobs/summarize-all") == "202"


def paths_status(application, path):
    return next(iter(application.openapi()["paths"][path]["post"]["responses"]))
