"""Desearch (Bittensor SN22) integration: normalization, credentials, wiring.

Offline — no network, no API keys. HTTP is monkeypatched/faked at the
request-helper level, mirroring the suite's FakeLLM discipline.
"""

from __future__ import annotations

import pytest

from deep_research.integration import validate_credentials
from deep_research.models import Credentials
from deep_research.run_config import RunConfig
from deep_research.utils import (
    _normalize_search_item,
    desearch_search_multiple,
    deduplicate_search_results,
)

_DS = Credentials({"DEEPSEEK_API_KEY": "x", "EXA_API_KEY": "y"})


# ── Normalization ─────────────────────────────────────────────────────────

def test_normalize_desearch_item_maps_link_to_url():
    item = {"title": "T", "snippet": "S", "link": "https://example.com/a"}
    out = _normalize_search_item(item, provider="desearch")
    assert out["url"] == "https://example.com/a"
    assert out["title"] == "T"
    assert out["content"] == "S"   # snippet is the preview + full-text fallback


def test_desearch_search_multiple_adapts_response(monkeypatch):
    captured = {}

    async def fake_request(method, path, *, params=None, json_body=None, timeout=None):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = json_body
        return {
            "search_results": [
                {"title": "A", "snippet": "sa", "link": "https://a.example"},
                {"title": "B", "snippet": "sb", "link": "https://b.example"},
            ],
        }, {"cost_usd": "0.001"}

    monkeypatch.setattr("deep_research.utils._desearch_request", fake_request)
    import asyncio
    responses = asyncio.run(desearch_search_multiple(["q"], max_results=5))
    assert captured["path"] == "/desearch/ai/search"
    assert captured["method"] == "POST"
    assert captured["body"]["tools"] == ["web", "twitter", "reddit"]
    assert captured["body"]["result_type"] == "ONLY_LINKS"
    assert len(responses) == 1
    results = responses[0]["results"]
    assert {r["url"] for r in results} == {"https://a.example", "https://b.example"}
    unique = deduplicate_search_results(responses)
    assert unique["https://a.example"]["content"] == "sa"


def test_desearch_search_multiple_respects_tool_and_date_filter(monkeypatch):
    captured = {}

    async def fake_request(method, path, *, params=None, json_body=None, timeout=None):
        captured["body"] = json_body
        return {"search_results": []}, {}

    monkeypatch.setattr("deep_research.utils._desearch_request", fake_request)
    import asyncio
    asyncio.run(desearch_search_multiple(
        ["q"], tools=["arxiv"], date_filter="PAST_WEEK"))
    assert captured["body"]["tools"] == ["arxiv"]
    assert captured["body"]["date_filter"] == "PAST_WEEK"


def test_desearch_search_multiple_swallows_errors(monkeypatch):
    async def fake_request(method, path, *, params=None, json_body=None, timeout=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("deep_research.utils._desearch_request", fake_request)
    import asyncio
    responses = asyncio.run(desearch_search_multiple(["q"]))
    assert responses == [{"results": []}]


def test_desearch_web_tool_registered():
    from deep_research.agents.web.tools import desearch_search
    assert desearch_search.name == "desearch_search"
    # tools/date_filter are model-visible args (not injected)
    schema = desearch_search.tool_call_schema.model_fields if \
        hasattr(desearch_search.tool_call_schema, "model_fields") else {}
    if schema:
        assert "tools" in schema
        assert "date_filter" in schema


# ── Credentials ───────────────────────────────────────────────────────────

def test_validate_engine_desearch_requires_desearch_key():
    cfg = RunConfig(web_search_engine="desearch")
    check = validate_credentials(config=cfg, credentials=_DS)
    assert "DESEARCH_API_KEY" in check.missing_required
    assert "EXA_API_KEY" not in check.missing_required


def test_validate_engine_desearch_satisfied():
    creds = Credentials({"DEEPSEEK_API_KEY": "x", "DESEARCH_API_KEY": "y"})
    cfg = RunConfig(web_search_engine="desearch",
                    supervisor_model_fallback_chain=["deepseek-v4.1-flash-1017"],
                    subagent_model_fallback_chain=["deepseek-v4.1-flash-1017"],
                    draft_report_model_fallback_chain=["deepseek-v4.1-flash-1017"])
    assert validate_credentials(config=cfg, credentials=creds).ok


def test_validate_engine_both_includes_desearch_key():
    cfg = RunConfig(web_search_engine="both")
    check = validate_credentials(config=cfg, credentials=_DS)
    assert "TAVILY_API_KEY" in check.missing_required
    assert "DESEARCH_API_KEY" in check.missing_required


def test_validate_reddit_with_desearch_key_makes_reddit_creds_optional():
    creds = Credentials({"DEEPSEEK_API_KEY": "x", "DESEARCH_API_KEY": "y"})
    cfg = RunConfig(enabled_agents=["ResearchWeb", "ResearchReddit"],
                    web_search_engine="desearch",
                    supervisor_model_fallback_chain=["deepseek-v4.1-flash-1017"],
                    subagent_model_fallback_chain=["deepseek-v4.1-flash-1017"],
                    draft_report_model_fallback_chain=["deepseek-v4.1-flash-1017"])
    check = validate_credentials(config=cfg, credentials=creds)
    for key in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
                "REDDIT_USERNAME", "REDDIT_PASSWORD"):
        assert key not in check.missing_required
    assert any("REDDIT" in name for name in check.optional_missing)


def test_validate_reddit_without_desearch_still_requires_reddit_creds():
    cfg = RunConfig(enabled_agents=["ResearchWeb", "ResearchReddit"])
    check = validate_credentials(config=cfg, credentials=_DS)
    for key in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
                "REDDIT_USERNAME", "REDDIT_PASSWORD"):
        assert key in check.missing_required


# ── Tool wiring ───────────────────────────────────────────────────────────

def test_search_tool_names_include_desearch():
    from deep_research.agents.base import SEARCH_TOOL_NAMES, SEARCH_INDEXABLE_TOOLS
    assert "desearch_search" in SEARCH_TOOL_NAMES
    assert "desearch_search" in SEARCH_INDEXABLE_TOOLS


def test_runtime_search_tools_include_desearch(monkeypatch):
    from deep_research.agents.base import _runtime_search_tools
    from deep_research.observer import Observer
    from deep_research.runtime import RuntimeContext, runtime_scope

    cfg = RunConfig(web_search_engine="desearch").finalize()
    ctx = RuntimeContext(
        run_id="r", thread_id="t", config=cfg,
        credentials=Credentials({"DESEARCH_API_KEY": "k"}),
        observer=Observer(run_id="r", config=cfg),
    )
    with runtime_scope(ctx):
        tools = _runtime_search_tools()
    names = [t.name for t in tools]
    assert names == ["desearch_search"]
