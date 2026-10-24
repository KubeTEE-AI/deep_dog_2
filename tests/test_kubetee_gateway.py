"""KubeTEE TEE gateway (llm.kubetee.ai) model routing.

Offline — no network. Guards the load-bearing invariants:
- "kubetee/*" names route to the gateway BEFORE name-based provider detection
  ("kubetee/z-ai/glm-5.3-flash" must never reach api.z.ai)
- the "kubetee/" prefix is a local routing key and is STRIPPED before the call
  (the gateway rejects the prefixed form — verified live 2026-10-24)
- key discovery / validate_credentials know the KubeTEE key aliases
"""

from __future__ import annotations

import pytest

from deep_research import config as _cfg
from deep_research.integration import validate_credentials
from deep_research.models import Credentials
from deep_research.run_config import RunConfig


def _model_name(m) -> str:
    return getattr(m, "model_name", None) or getattr(m, "model", None)

def _base_url(m) -> str:
    return getattr(m, "openai_api_base", None) or getattr(m, "base_url", None)


# ── Routing + prefix stripping ────────────────────────────────────────────

def test_kubetee_name_routes_to_gateway_and_strips_prefix(monkeypatch):
    monkeypatch.setenv("KUBETEE_API_KEY", "test-key")
    m = _cfg.get_model("kubetee/z-ai/glm-5.3-flash")
    assert _model_name(m) == "z-ai/glm-5.3-flash"   # prefix stripped
    assert "llm.kubetee.ai" in (_base_url(m) or "")  # not api.z.ai


def test_kubetee_deepseek_name_does_not_leak_to_deepseek_api(monkeypatch):
    monkeypatch.setenv("KUBETEE_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "native-key")  # must NOT be used
    m = _cfg.get_model("kubetee/deepseek/deepseek-v4.1-flash")
    assert _model_name(m) == "deepseek/deepseek-v4.1-flash"
    assert "llm.kubetee.ai" in (_base_url(m) or "")
    assert "deepseek.com" not in (_base_url(m) or "")


def test_bare_glm_name_still_routes_to_zai(monkeypatch):
    """Regression: the kubetee/ prefix check must not disturb name detection."""
    monkeypatch.setenv("ZHIPUAI_API_KEY", "z-key")
    m = _cfg.get_model("glm-5.3-flash")
    assert "z.ai" in (_base_url(m) or "")


def test_kubetee_api_base_env_override(monkeypatch):
    monkeypatch.setenv("KUBETEE_API_KEY", "test-key")
    monkeypatch.setenv("KUBETEE_API_BASE", "https://kubetee-proxy.internal/v1")
    m = _cfg.get_model("kubetee/ornith/ornith-1.5-397b")
    assert _model_name(m) == "ornith/ornith-1.5-397b"
    assert _base_url(m) == "https://kubetee-proxy.internal/v1"


def test_kubetee_prefix_only_is_rejected():
    with pytest.raises(ValueError, match="kubetee/"):
        _cfg.get_model("kubetee/")


# ── Key discovery ─────────────────────────────────────────────────────────

def test_model_required_keys_kubetee_aliases():
    keys = _cfg.model_required_keys("kubetee/z-ai/glm-5.3-flash")
    assert "KUBETEE_API_KEY" in keys
    assert "KUBETEE_API_TOKEN" in keys
    assert "LITELLM_API_KEY" in keys
    # must NOT report the native provider's key
    assert "ZHIPUAI_API_KEY" not in keys


def test_model_required_keys_native_names_unchanged():
    assert _cfg.model_required_keys("glm-5.3-flash") == ["ZHIPUAI_API_KEY"]
    assert "DEEPSEEK_API_KEY" in _cfg.model_required_keys("deepseek-v4.1-flash-1017")


def test_validate_credentials_kubetee_key_required(monkeypatch):
    # Hermetic: Credentials.has() falls back to the process env, and the suite
    # may have .env values loaded (run_research.py calls load_dotenv).
    for name in ("KUBETEE_API_KEY", "KUBETEE_API_TOKEN", "LITELLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = RunConfig(supervisor_model_fallback_chain=["kubetee/z-ai/glm-5.3-flash"],
                    subagent_model_fallback_chain=["kubetee/z-ai/glm-5.3-flash"],
                    draft_report_model_fallback_chain=["kubetee/z-ai/glm-5.3-flash"],
                    web_search_engine="exa")
    creds = Credentials({"EXA_API_KEY": "y"})   # no KubeTEE key
    check = validate_credentials(config=cfg, credentials=creds)
    assert "KUBETEE_API_KEY" in check.missing_required


def test_validate_credentials_kubetee_key_satisfied():
    cfg = RunConfig(supervisor_model_fallback_chain=["kubetee/deepseek/deepseek-v4.1-flash"],
                    subagent_model_fallback_chain=["kubetee/deepseek/deepseek-v4.1-flash"],
                    draft_report_model_fallback_chain=["kubetee/deepseek/deepseek-v4.1-flash"],
                    web_search_engine="exa")
    creds = Credentials({"KUBETEE_API_KEY": "k", "EXA_API_KEY": "y"})
    assert validate_credentials(config=cfg, credentials=creds).ok


def test_model_chain_filtering_knows_kubetee_names(monkeypatch):
    monkeypatch.delenv("KUBETEE_API_KEY", raising=False)
    monkeypatch.delenv("KUBETEE_API_TOKEN", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    chain = ["kubetee/z-ai/glm-5.3-flash"]
    assert _cfg._filter_usable_chain(chain, False, "test") == []
    monkeypatch.setenv("KUBETEE_API_KEY", "k")
    assert _cfg._filter_usable_chain(chain, False, "test") == chain
