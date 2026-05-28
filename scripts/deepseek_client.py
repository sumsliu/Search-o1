from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING_ENABLED,
    get_deepseek_model,
    require_deepseek_api_key,
)

_client: Optional[OpenAI] = None


def get_deepseek_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=require_deepseek_api_key(),
            base_url=DEEPSEEK_BASE_URL,
        )
    return _client


def deepseek_chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    variant: Optional[str] = None,
    thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[List[str], str]] = None,
    **kwargs: Any,
):
    """Call DeepSeek V4 chat completions with optional thinking mode."""
    client = get_deepseek_client()
    model_id = model or get_deepseek_model(variant)
    use_thinking = DEEPSEEK_THINKING_ENABLED if thinking is None else thinking
    effort = reasoning_effort or DEEPSEEK_REASONING_EFFORT

    request_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    if stop is not None:
        request_kwargs["stop"] = stop
    request_kwargs.update(kwargs)

    if use_thinking:
        request_kwargs["reasoning_effort"] = effort
        request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    return client.chat.completions.create(**request_kwargs)


def message_to_text(message, include_reasoning: bool = True) -> str:
    reasoning = getattr(message, "reasoning_content", None) or ""
    content = message.content or ""
    if include_reasoning and reasoning:
        return f"{reasoning}{content}"
    return content


def deepseek_chat_text(
    messages: List[Dict[str, str]],
    *,
    include_reasoning: bool = False,
    **kwargs: Any,
) -> str:
    """Return assistant text; optionally prepend reasoning_content."""
    response = deepseek_chat(messages, **kwargs)
    message = response.choices[0].message
    return message_to_text(message, include_reasoning=include_reasoning)
