import os
import sys
from pathlib import Path
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner

# Allow importing project-level env config when running from scripts/
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (  # noqa: E402
    DEEPSEEK_BASE_URL,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING_ENABLED,
    get_deepseek_model,
    require_deepseek_api_key,
)


class DeepSeekRunner(BaseRunner):
    client = OpenAI(
        api_key=require_deepseek_api_key(),
        base_url=DEEPSEEK_BASE_URL,
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        model_name = args.model or get_deepseek_model()
        if model_name.lower() in ("flash", "pro"):
            model_name = get_deepseek_model(model_name)

        self.client_kwargs: dict[str | str] = {
            "model": model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "timeout": args.openai_timeout,
            # "stop": args.stop, --> stop is only used for base models currently
        }
        if DEEPSEEK_THINKING_ENABLED:
            self.client_kwargs["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
            self.client_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        def __run_single(counter):
            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                content = response.choices[0].message.content
                return content
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as e:
                print("Exception: ", repr(e))
                print("Sleeping for 30 seconds...")
                print("Consider reducing the number of parallel processes.")
                sleep(30)
                return DeepSeekRunner._run_single(prompt)
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e
        return outputs
