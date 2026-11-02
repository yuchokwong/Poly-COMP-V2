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


def test_query_length_is_capped(client):
    long_query = "a" * 201
    assert (
        client.get("/api/strategies/clip/search", params={"q": long_query}).status_code == 422
    )
    assert client.get("/api/search", params={"q": long_query}).status_code == 422
    assert client.get("/api/compare", params={"q": long_query}).status_code == 422

    allowed = client.get("/api/strategies/clip/search", params={"q": "a" * 200})
    assert allowed.status_code == 503  # passes validation, strategy is not built


def test_images_are_cacheable(cfg, store):
    from PIL import Image

    images = Path(cfg.paths.data_dir) / "images" / "memecap"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 30, 30)).save(images / "memes_00000.jpg", "JPEG")

    with TestClient(create_app(cfg)) as client:
        for url in ("/api/image/1", "/api/thumbnail/1?size=64"):
            response = client.get(url)
            assert response.status_code == 200, url
            assert response.headers["cache-control"] == "public, max-age=86400"


def test_api_docs_are_closed_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_api_docs_can_be_enabled(cfg):
    cfg.app.expose_docs = True
    with TestClient(create_app(cfg)) as client:
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200


def test_settings_put_requires_the_admin_token(cfg):
    cfg.app.admin_token = "s3cret"
    with TestClient(create_app(cfg)) as client:
        listing = client.get("/api/settings")
        assert listing.status_code == 200
        assert listing.json()["read_only"] is True

        denied = client.put("/api/settings", json={"app.default_top_k": 20})
        assert denied.status_code == 403

        wrong = client.put(
            "/api/settings",
            json={"app.default_top_k": 20},
            headers={"X-Admin-Token": "nope"},
        )
        assert wrong.status_code == 403

        ok = client.put(
            "/api/settings",
            json={"app.default_top_k": 20},
            headers={"X-Admin-Token": "s3cret"},
        )
        assert ok.status_code == 200
        assert cfg.app.default_top_k == 20


def test_settings_page_and_script_are_served(client):
    page = client.get("/settings")
    assert page.status_code == 200
    assert "settings.js" in page.text

    script = client.get("/static/settings.js")
    assert script.status_code == 200
    assert "/api/settings" in script.text
