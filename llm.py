"""A tiny client for an OpenAI-compatible chat API (TokenRouter by default). Standard library only.

Settings come from the environment or a `.env` file in this folder or the folder above it:

    TOKENROUTER_BASE_URL=https://api.tokenrouter.com/v1
    TOKENROUTER_API_KEY=...
    SNAKE_AI_LLM_MODEL=anthropic/claude-haiku-5.5    # optional

`.env` is git-ignored: keys never end up in the repository.
"""
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"


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


def settings():
    load_env()
    return {"base_url": os.environ.get("TOKENROUTER_BASE_URL", "").rstrip("/"),
            "api_key": os.environ.get("TOKENROUTER_API_KEY", ""),
            "model": os.environ.get("SNAKE_AI_LLM_MODEL") or DEFAULT_MODEL}


def status():
    """What the dashboard may show: whether it is set up and which model it uses (never the key)."""
    s = settings()
    return {"configured": bool(s["base_url"] and s["api_key"]), "model": s["model"]}


def chat(messages, max_tokens=4000, temperature=None, json_mode=True, timeout=180, retries=2):
    """Send a chat; returns (text, usage). Raises LLMError with a readable message."""
    s = settings()
    if not (s["base_url"] and s["api_key"]):
        raise LLMError("No LLM configured: set TOKENROUTER_BASE_URL and TOKENROUTER_API_KEY in .env")
    body = {"model": s["model"], "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:  # some models (e.g. Claude Opus 5.5) refuse it
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(s["base_url"] + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {s['api_key']}",
                                          "Content-Type": "application/json"})
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
            raise LLMError(f"LLM request failed ({e.code}): {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            raise LLMError(f"Could not reach the LLM: {e}") from None
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
    print(f"configured: {st['configured']}  model: {st['model']}")
    if st["configured"]:
        print(chat_json([{"role": "user", "content": 'Reply with {"ok": true}'}], max_tokens=50))
