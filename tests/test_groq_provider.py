"""Tests cho Groq provider.

Groq dùng OpenAI-compatible API nên test monkeypatch ``OpenAI`` tương tự
như OpenAI branch, nhưng với ``base_url='https://api.groq.com/openai/v1'``
và key đọc từ ``GROQ_API_KEY``.

Các test trong file này mutate ``sys.modules['openai']`` nên phải restore
lại module thật + reload các module src phụ thuộc sau mỗi test. Nếu không
restore, các test khác (vd. ``test_hyde_fallback_uses_bigrams``) sẽ thấy
fake client bị leak qua và fail.
"""

from __future__ import annotations

import importlib
import sys as _sys

import pytest


def _install_fake_openai(request, FakeClient):
    """Patch ``openai`` module và đăng ký restore trên teardown."""
    real_openai = _sys.modules.get("openai")
    fake_mod = type(_sys)("fake_openai")
    fake_mod.OpenAI = FakeClient
    _sys.modules["openai"] = fake_mod
    for modname in (
        "src.task10_generation",
        "src.bonus5_conversation_memory",
        "src.bonus3_query_expansion",
    ):
        if modname in _sys.modules:
            importlib.reload(_sys.modules[modname])

    def _restore():
        if real_openai is None:
            _sys.modules.pop("openai", None)
        else:
            _sys.modules["openai"] = real_openai
        for modname in (
            "src.task10_generation",
            "src.bonus5_conversation_memory",
            "src.bonus3_query_expansion",
        ):
            if modname in _sys.modules:
                importlib.reload(_sys.modules[modname])

    request.addfinalizer(_restore)


@pytest.fixture(autouse=True)
def _reload_pipeline_after_test(request):
    """Reload các src module có thể đã bị fake trong test khác."""
    yield
    for modname in (
        "src.task10_generation",
        "src.bonus5_conversation_memory",
        "src.bonus3_query_expansion",
        "src.task9_retrieval_pipeline",
    ):
        try:
            if modname in _sys.modules:
                importlib.reload(_sys.modules[modname])
        except Exception:
            pass


def _captured(request, monkeypatch, *, response_text: str = "Groq says [1] is correct."):
    """Patch OpenAI client để capture base_url + model và trả text."""
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            class _Choice:
                class _Msg:
                    content = response_text
                message = _Msg()
            return type("R", (), {"choices": [_Choice()]})()

    class FakeClient:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw

        class chat:
            completions = FakeCompletions()

    _install_fake_openai(request, FakeClient)
    import src.task10_generation as gen
    monkeypatch.setattr(gen, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(gen, "LLM_MODEL", "")
    monkeypatch.setattr("os.environ", {
        **__import__("os").environ,
        "GROQ_API_KEY": "test-groq-key",
        "OPENAI_API_KEY": "",
        "LLM_PROVIDER": "groq",
    })
    return captured


def test_groq_uses_base_url_and_groq_key(request, monkeypatch):
    captured = _captured(request, monkeypatch)
    from src.task10_generation import call_llm
    out = call_llm("system", "user")
    assert "Groq says" in out
    assert captured["client_kwargs"]["api_key"] == "test-groq-key"
    assert captured["client_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"
    # Default model Groq mặc định: qwen/qwen3.8-27b (chat model phổ biến
    # trên hầu hết account, kể cả free tier; hỗ trợ tiếng Việt).
    # Override qua LLM_MODEL nếu cần.
    assert captured["model"] == "qwen/qwen3.8-27b"


def test_groq_uses_explicit_llm_model(request, monkeypatch):
    captured = _captured(request, monkeypatch)
    import src.task10_generation as gen
    gen.LLM_MODEL = "llama-3.3-70b-versatile"
    from src.task10_generation import call_llm
    call_llm("system", "user")
    assert captured["model"] == "llama-3.3-70b-versatile"


def test_groq_returns_empty_when_no_key(monkeypatch):
    import src.task10_generation as gen
    monkeypatch.setattr(gen, "LLM_PROVIDER", "groq")
    monkeypatch.setattr("os.environ", {
        **__import__("os").environ,
        "GROQ_API_KEY": "",
        "OPENAI_API_KEY": "",
    })
    from src.task10_generation import call_llm
    assert call_llm("system", "user") == ""


def test_groq_returns_empty_on_provider_exception(request, monkeypatch):
    import src.task10_generation as gen
    monkeypatch.setattr(gen, "LLM_PROVIDER", "groq")
    monkeypatch.setattr("os.environ", {
        **__import__("os").environ,
        "GROQ_API_KEY": "x",
        "OPENAI_API_KEY": "",
    })

    class BoomClient:
        def __init__(self, **kw):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("groq down")

    _install_fake_openai(request, BoomClient)
    from src.task10_generation import call_llm
    assert call_llm("system", "user") == ""


def test_bonus5_llm_rewrite_uses_groq(request, monkeypatch):
    """Conversation memory với use_llm=True phải đi qua Groq khi provider=groq."""
    from src.bonus5_conversation_memory import _try_llm_rewrite

    captured: dict = {}
    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            class C:
                class M:
                    content = "IELTS là gì?"
                message = M()
            return type("R", (), {"choices": [C()]})()
    class FakeClient:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw
        class chat:
            completions = FakeCompletions()

    _install_fake_openai(request, FakeClient)
    monkeypatch.setattr("os.environ", {
        **__import__("os").environ,
        "GROQ_API_KEY": "groq-test",
        "OPENAI_API_KEY": "",
        "LLM_PROVIDER": "groq",
    })
    history = [{"role": "user", "content": "IELTS có được dùng để xét tuyển không?"}]
    out = _try_llm_rewrite(history, "vậy hả?")
    assert out == "IELTS là gì?"
    assert captured["client_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"


def test_bonus3_hyde_via_groq(request, monkeypatch):
    """Bonus 3 HyDE cũng qua Groq khi provider=groq."""
    from src.bonus3_query_expansion import _hyde_via_llm

    captured: dict = {}
    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            class C:
                class M:
                    content = "Hypothetical passage about IELTS..."
                message = M()
            return type("R", (), {"choices": [C()]})()
    class FakeClient:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw
        class chat:
            completions = FakeCompletions()

    _install_fake_openai(request, FakeClient)
    monkeypatch.setattr("os.environ", {
        **__import__("os").environ,
        "GROQ_API_KEY": "groq-test",
        "OPENAI_API_KEY": "",
        "LLM_PROVIDER": "groq",
    })
    out = _hyde_via_llm("IELTS xét tuyển")
    assert "Hypothetical passage" in out
    assert captured["client_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"
