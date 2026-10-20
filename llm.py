"""A tiny client for an OpenAI-compatible chat API: TokenRouter or OpenRouter. Standard library only.

Settings come from the environment or a `.env` file in this folder or the folder above it. Set the key of
either provider (or both):

    TOKENROUTER_API_KEY=...            # https://tokenrouter.com
    OPENROUTER_API_KEY=...             # https://openrouter.ai
    SNAKE_AI_LLM_PROVIDER=openrouter   # optional: which one to use when both have a key (default tokenrouter)
    SNAKE_AI_LLM_MODEL=anthropic/claude-haiku-5.5    # optional; both providers use this model id
    TOKENROUTER_BASE_URL / OPENROUTER_BASE_URL       # optional: another endpoint of that provider

Any other OpenAI-compatible API works as the "tokenrouter" provider with its base URL and key.
`.env` is git-ignored: keys never end up in the repository.
"""
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
PROVIDERS = {
    "tokenrouter": {"label": "TokenRouter", "base_url": "https://api.tokenrouter.com/v1",
                    "key": "TOKENROUTER_API_KEY", "url": "TOKENROUTER_BASE_URL", "headers": {}},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
                   "key": "OPENROUTER_API_KEY", "url": "OPENROUTER_BASE_URL",
                   "headers": {"X-Title": "Snake AI lab"}},  # (OpenRouter's optional app attribution)
}
SETUP_HINT = "add TOKENROUTER_API_KEY or OPENROUTER_API_KEY to .env (see .env.example)"


class LLMError(RuntimeError):
    pass


def load_env():
    """Read KEY=VALUE lines from .env files into os.environ (real environment variables win)."""
    for path in (os.path.join(ROOT, ".env"), os.path.join(os.path.dirname(ROOT), ".env")):
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def provider_name():
    """SNAKE_AI_LLM_PROVIDER if set, else the first provider with a key (TokenRouter, then OpenRouter)."""
    load_env()
    chosen = os.environ.get("SNAKE_AI_LLM_PROVIDER", "").strip().lower()
    if chosen:
        if chosen not in PROVIDERS:
            raise LLMError(f"SNAKE_AI_LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, not {chosen!r}")
        return chosen
    return next((n for n, p in PROVIDERS.items() if os.environ.get(p["key"])), "tokenrouter")


def settings():
    name = provider_name()
    p = PROVIDERS[name]
    return {"provider": name, "label": p["label"],
            "base_url": (os.environ.get(p["url"]) or p["base_url"]).rstrip("/"),
            "api_key": os.environ.get(p["key"], ""), "headers": p["headers"],
            "model": os.environ.get("SNAKE_AI_LLM_MODEL") or DEFAULT_MODEL}


def status():
    """What the dashboard may show: whether it is set up, the provider and the model (never the key)."""
    try:
        s = settings()
    except LLMError as e:
        return {"configured": False, "model": DEFAULT_MODEL, "provider": None, "error": str(e), "hint": SETUP_HINT}
    return {"configured": bool(s["base_url"] and s["api_key"]), "model": s["model"], "provider": s["label"],
            "hint": SETUP_HINT}


def chat(messages, max_tokens=4000, temperature=None, json_mode=True, timeout=180, retries=2):
    """Send a chat; returns (text, usage). Raises LLMError with a readable message."""
    s = settings()
    if not (s["base_url"] and s["api_key"]):
        raise LLMError(f"No LLM configured: {SETUP_HINT}")
    body = {"model": s["model"], "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:  # some models (e.g. Claude Opus 5.5) refuse it
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(s["base_url"] + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {s['api_key']}",
                                          "Content-Type": "application/json", **s["headers"]})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                data = json.loads(res.read())
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise LLMError(f"{s['label']} request failed ({e.code}): {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise LLMError(f"Could not reach {s['label']}: {e}") from None
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"Unexpected LLM response: {str(data)[:300]}") from None
    if choice.get("finish_reason") == "length":
        raise LLMError("The LLM's answer was cut off (max_tokens too small)")
    usage = data.get("usage") or {}
    return text, {"model": data.get("model", s["model"]), "prompt_tokens": usage.get("prompt_tokens"),
                  "completion_tokens": usage.get("completion_tokens")}


def chat_json(messages, **kw):
    """chat() that parses the answer as a JSON object (tolerates a ```json fence around it)."""
    text, usage = chat(messages, **kw)
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        obj = json.loads(t)
    except ValueError:
        start, end = t.find("{"), t.rfind("}")
        try:
            obj = json.loads(t[start:end + 1])
        except ValueError:
            raise LLMError(f"The LLM did not return JSON: {text[:200]}") from None
    if not isinstance(obj, dict):
        raise LLMError("The LLM returned JSON that is not an object")
    return obj, usage


if __name__ == "__main__":
    st = status()
    print(f"configured: {st['configured']}  provider: {st['provider']}  model: {st['model']}")
    if st["configured"]:
        print(chat_json([{"role": "user", "content": 'Reply with {"ok": true}'}], max_tokens=50))
