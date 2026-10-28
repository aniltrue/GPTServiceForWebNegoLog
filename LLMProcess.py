import asyncio
import json
import os
import random
from typing import Optional, Any, Dict

from dotenv import load_dotenv

from openai import AsyncOpenAI, RateLimitError, APIStatusError, BadRequestError

from AsyncLogger import AsyncLogger
from Models import RequestModel

load_dotenv(dotenv_path='service.env')
load_dotenv(dotenv_path='.env')

MAX_RETRY = int(os.getenv("MAX_RETRY", 10))
RATE_MAX_RETRY = int(os.getenv("RATE_MAX_RETRY", 10))
RATE_BACKOFF_BASE = float(os.getenv("RATE_BACKOFF_BASE", 1.0))
RATE_BACKOFF_MAX = float(os.getenv("RATE_BACKOFF_MAX", 10.0))

DEFAULT_REASONING_EFFORT = os.getenv("DEFAULT_REASONING_EFFORT", "low")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.2))

# Parameters the model may reject (e.g. temperature on some reasoning models); they are dropped and retried
OPTIONAL_PARAMS = ("temperature", "reasoning_effort")


class LLMProcessor:
    def __init__(self, logger: AsyncLogger) -> None:
        self.logger = logger

        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

        self._usage_lock = asyncio.Lock()

    async def _update_usage(self, usage: Optional[object]) -> None:
        if usage is None:
            return

        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or 0

        async with self._usage_lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total

    async def _wait_limit(self, status_code: int, rate_limit_counter: int) -> int:
        delay = min(
            RATE_BACKOFF_BASE * (2 ** (rate_limit_counter)) +
            random.uniform(0, 0.5),
            RATE_BACKOFF_MAX,
        )

        await self.logger.info(
            f"[API {status_code}] Rate limit hit (attempt {rate_limit_counter}/{RATE_MAX_RETRY}), "
            f"backing off for {delay:.2f}s")

        rate_limit_counter += 1

        await asyncio.sleep(delay)

        return rate_limit_counter

    async def run(self, request: RequestModel) -> Dict[str, Any]:
        rate_limit_counter = 0
        validation_counter = 0

        messages = []

        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.user_prompt})

        params = {
            "reasoning_effort": request.reasoning_effort or DEFAULT_REASONING_EFFORT,
            "temperature": request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE,
        }

        while validation_counter <= MAX_RETRY and rate_limit_counter <= RATE_MAX_RETRY:
            try:
                completion = await self.client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    n=1,
                    **params
                )

                result = completion.choices[0].message.content

                await self._update_usage(completion.usage)

                return json.loads(result)
            except RateLimitError as e:
                rate_limit_counter = await self._wait_limit(429, rate_limit_counter)

            except BadRequestError as e:
                if e.param in OPTIONAL_PARAMS and e.param in params:
                    await self.logger.warning(f"{request.model} does not support `{e.param}`, retrying without it.")

                    params.pop(e.param)
                else:
                    raise
            except APIStatusError as e:
                if e.status_code in (429, 500, 502, 503, 504):
                    rate_limit_counter = await self._wait_limit(e.status_code, rate_limit_counter)
                else:
                    raise
            except ValueError as e:
                await self.logger.warning(f"Validation error! Try {validation_counter}/{MAX_RETRY}")

                messages.append({"role": "assistant", "content": str(result)})

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer did not match the JSON schema. "
                            f"\nError summary: {e}. "
                            "\nPlease answer again, returning ONLY a valid JSON object."
                        ),
                    }
                )

                validation_counter += 1
            except Exception:
                raise

        if rate_limit_counter > RATE_MAX_RETRY:
            raise Exception("Reaching max. retrying for Rate Limit")

        if validation_counter > MAX_RETRY:
            raise Exception("Reaching max. retrying for Validation")

        return {}
