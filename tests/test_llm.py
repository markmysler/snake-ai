"""LLM provider selection: TokenRouter or OpenRouter, from the environment (keys are never reported)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import llm  # noqa: E402

KEYS = ("TOKENROUTER_API_KEY", "OPENROUTER_API_KEY", "TOKENROUTER_BASE_URL", "OPENROUTER_BASE_URL",
        "SNAKE_AI_LLM_PROVIDER", "SNAKE_AI_LLM_MODEL")


@pytest.fixture
def env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "load_env", lambda: None)  # ignore a real .env
    return monkeypatch


def test_nothing_configured(env):
    st = llm.status()
    assert not st["configured"] and "OPENROUTER_API_KEY" in st["hint"]
    with pytest.raises(llm.LLMError, match="No LLM configured"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_picks_the_provider_with_a_key(env):
    env.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    s = llm.settings()
    assert s["provider"] == "openrouter" and s["base_url"] == "https://openrouter.ai/api/v1"
    st = llm.status()
    assert st == {"configured": True, "model": llm.DEFAULT_MODEL, "provider": "OpenRouter", "hint": llm.SETUP_HINT}
    assert "secret" not in json.dumps(st)
    env.setenv("TOKENROUTER_API_KEY", "tr-secret")  # both: TokenRouter unless chosen
    assert llm.settings()["provider"] == "tokenrouter"
    env.setenv("SNAKE_AI_LLM_PROVIDER", "OpenRouter")
    assert llm.settings()["provider"] == "openrouter"
    env.setenv("SNAKE_AI_LLM_PROVIDER", "nope")
    assert not llm.status()["configured"]


def test_request_goes_to_the_chosen_provider(env):
    env.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    env.setenv("SNAKE_AI_LLM_MODEL", "anthropic/claude-sonnet-5.5")
    sent = {}

    class Res:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]}).encode()

    def fake_urlopen(req, timeout):
        sent.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data))
        return Res()

    env.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    obj, _ = llm.chat_json([{"role": "user", "content": "hi"}])
    assert obj == {"ok": True}
    assert sent["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer sk-or-secret" and sent["headers"]["X-title"] == "Snake AI lab"
    assert sent["body"]["model"] == "anthropic/claude-sonnet-5.5"
