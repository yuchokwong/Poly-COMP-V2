from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tsr.api import create_app  # noqa: E402


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg)) as test_client:
        yield test_client


def test_health_and_stats_without_index(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "degraded"
    assert body["ready"] is False
    assert body["num_images"] == 0
    assert {item["key"] for item in body["strategies"]} == {"clip", "clip_nlp", "vlm"}

    stats = client.get("/api/stats")
    assert stats.status_code == 200
    payload = stats.json()
    assert payload["dataset"]["name"] == "memecap"
    assert payload["dataset"]["num_images"] == 0
    assert payload["dataset"]["manifest_exists"] is False
    assert payload["models"]["clip"].startswith("ViT-")
    assert payload["store_error"] is None


def test_strategy_listing_and_per_strategy_stats(client):
    listing = client.get("/api/strategies")
    assert listing.status_code == 200
    items = listing.json()
    assert [item["key"] for item in items] == ["clip", "clip_nlp", "vlm"]
    assert all(item["ready"] is False for item in items)
    assert all(item["error"] for item in items)

    one = client.get("/api/strategies/clip_nlp/stats")
    assert one.status_code == 200
    assert one.json()["key"] == "clip_nlp"
    assert one.json()["ready"] is False


def test_search_errors(client):
    missing = client.get("/api/strategies/nope/search", params={"q": "cat"})
    assert missing.status_code == 404

    unavailable = client.get("/api/strategies/clip/search", params={"q": "cat"})
    assert unavailable.status_code == 503

    legacy = client.get("/api/search", params={"q": "cat"})
    assert legacy.status_code == 503

    unknown_legacy = client.get("/api/search", params={"q": "cat", "strategy": "nope"})
    assert unknown_legacy.status_code == 404


def test_compare_reports_unavailable_columns(client):
    response = client.get("/api/compare", params={"q": "a cat at dinner", "k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["columns"]) == {"clip", "clip_nlp", "vlm"}
    assert set(payload["errors"]) == {"clip", "clip_nlp", "vlm"}
    for column in payload["columns"].values():
        assert column["ready"] is False
        assert column["results"] == []
        assert column["count"] == 0


def test_images_detail_random_and_eval_endpoints(client):
    assert client.get("/api/image/1").status_code == 404
    assert client.get("/api/thumbnail/1").status_code == 404
    assert client.get("/api/detail/1").status_code == 404
    assert client.get("/api/random").json() == {"results": []}
    assert client.get("/api/eval/summary").status_code == 404
    assert client.get("/api/eval/report.html").status_code == 404

    queries = client.get("/api/eval/queries")
    assert queries.status_code == 200
    assert queries.json()["count"] == 0


def test_pages_are_served(client):
    for path in ("/", "/clip", "/clip-nlp", "/vlm", "/compare", "/evaluation"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "text/html" in response.headers["content-type"]


def test_settings_get_and_put(client, cfg):
    listing = client.get("/api/settings")
    assert listing.status_code == 200
    fields = {field["key"]: field for field in listing.json()["fields"]}
    assert fields["strategies.clip_nlp.weights.doc"]["editable"] is True
    assert fields["models.vlm.model"]["editable"] is False

    ok = client.put(
        "/api/settings",
        json={"app.default_top_k": 20, "strategies.vlm.clip_weight": 0.15},
    )
    assert ok.status_code == 200
    assert cfg.app.default_top_k == 20
    assert abs(cfg.strategies.vlm.clip_weight - 0.15) < 1e-9
    assert (Path(cfg.paths.data_dir) / "settings.json").exists()

    readonly = client.put("/api/settings", json={"models.vlm.model": "nope"})
    assert readonly.status_code == 422

    out_of_range = client.put("/api/settings", json={"strategies.vlm.clip_weight": 2})
    assert out_of_range.status_code == 422


def test_settings_page_and_script_are_served(client):
    page = client.get("/settings")
    assert page.status_code == 200
    assert "settings.js" in page.text

    script = client.get("/static/settings.js")
    assert script.status_code == 200
    assert "/api/settings" in script.text
