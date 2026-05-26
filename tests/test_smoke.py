"""
Smoke tests — verifies both LLM clients can complete a basic request.

WHY marked 'integration': these call REAL APIs and use REAL tokens.
They should NOT run on every test invocation. Run explicitly:
    pytest -m integration

In CI we run them on a separate workflow with secrets configured.
"""

import pytest

from harness.clients import GeminiClient, GroqClient, LLMClient, LLMRequest


@pytest.mark.integration
@pytest.mark.smoke
def test_groq_completes_basic_request() -> None:
    """Groq client returns a non-empty response with token counts and latency."""
    with GroqClient() as client:
        response = client.complete(
            LLMRequest(prompt="Reply with exactly the word: PONG", max_tokens=256)
        )

    assert response.text.strip(), "expected non-empty text"
    assert response.provider == "groq"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0
    assert "PONG" in response.text.upper()


@pytest.mark.integration
@pytest.mark.smoke
def test_gemini_completes_basic_request() -> None:
    """Gemini client returns a non-empty response with token counts and latency."""
    with GeminiClient() as client:
        response = client.complete(
            LLMRequest(prompt="Reply with exactly the word: PONG", max_tokens=256)
        )

    assert response.text.strip()
    assert response.provider == "gemini"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0
    assert "PONG" in response.text.upper()


@pytest.mark.integration
def test_both_clients_satisfy_the_protocol() -> None:
    """Structural check — both implementations conform to LLMClient.

    WHY this test matters: if someone changes the Protocol later, this
    test will fail at the SAME instant the implementations diverge.
    It's a contract test, not a behavior test.
    """
    groq: LLMClient = GroqClient()
    gemini: LLMClient = GeminiClient()

    assert groq.provider_name == "groq"
    assert gemini.provider_name == "gemini"

    # WHY this loop: same code path against different providers.
    # Demonstrates the abstraction is real, not theoretical.
    for client in (groq, gemini):
        response = client.complete(LLMRequest(prompt="Say hi", max_tokens=256))
        assert response.text
        assert response.provider in {"groq", "gemini"}