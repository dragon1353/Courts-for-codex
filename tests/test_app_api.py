import app as app_module
import pytest
from rag_agent import javascript_string


@pytest.fixture(autouse=True)
def reset_task_statuses():
    app_module.training_status = {"state": "idle", "message": "閒置"}
    app_module.analysis_status = {"state": "idle", "message": "閒置"}
    app_module.download_status = {"state": "idle", "message": "閒置"}


@pytest.fixture
def client():
    return app_module.app.test_client()


def test_download_rejects_invalid_and_reversed_dates(client):
    invalid = client.post(
        "/start_download",
        json={
            "start_year": "115",
            "start_month": "02",
            "start_day": "30",
            "end_year": "115",
            "end_month": "03",
            "end_day": "01",
        },
    )
    assert invalid.status_code == 400

    reversed_range = client.post(
        "/start_download",
        json={
            "start_year": "115",
            "start_month": "03",
            "start_day": "02",
            "end_year": "115",
            "end_month": "03",
            "end_day": "01",
        },
    )
    assert reversed_range.status_code == 400


def test_training_rejects_missing_folder(client):
    response = client.post(
        "/start_training", json={"folder_path": "Z:/folder-that-does-not-exist"}
    )
    assert response.status_code == 400


def test_busy_analysis_blocks_new_download(client):
    app_module.analysis_status = {"state": "analyzing", "message": "分析中"}
    response = client.post(
        "/start_download",
        json={
            "start_year": "115",
            "start_month": "03",
            "start_day": "01",
            "end_year": "115",
            "end_month": "03",
            "end_day": "02",
        },
    )
    assert response.status_code == 409


def test_rag_stream_updates_status_and_saves_report(monkeypatch, tmp_path):
    report_path = tmp_path / "final_report.html"
    monkeypatch.setattr(app_module, "FINAL_REPORT_FILE", str(report_path))
    monkeypatch.setattr(app_module, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(
        app_module, "query_rag_system", lambda _prompt: iter(["<div>", "完成", "</div>"])
    )

    chunks = list(app_module.generate_rag_stream("測試問題"))

    assert "".join(chunks) == "<div>完成</div>"
    assert report_path.read_text(encoding="utf-8") == "<div>完成</div>"
    assert app_module.analysis_status["state"] == "success"


def test_javascript_string_escapes_script_delimiters():
    escaped = javascript_string('</script><script>alert("x")</script>')
    assert "</script>" not in escaped
    assert "\\u003c" in escaped
