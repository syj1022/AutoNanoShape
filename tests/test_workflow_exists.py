from pathlib import Path


def test_publish_workflow_exists():
    assert Path(".github/workflows/publish.yml").exists()
