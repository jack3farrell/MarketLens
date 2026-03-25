"""Tests for the health check endpoint."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from marketlens.main import app

client = TestClient(app)


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_returns_ok():
    r = client.get("/health")
    assert r.json()["status"] == "ok"


def test_root_returns_200():
    r = client.get("/")
    assert r.status_code == 200


def test_root_contains_docs_link():
    r = client.get("/")
    body = r.json()
    assert "docs" in body