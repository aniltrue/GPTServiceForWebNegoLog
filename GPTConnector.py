import json
import uuid
from typing import Optional, Literal, Dict, Any

import anyio
import redis.asyncio as aioredis
from pydantic import BaseModel, Field


class ResponseModel(BaseModel):
    status: Literal['Success', 'Error'] = Field('Success', alias='Status')
    error_message: Optional[str] = Field(None, alias="ErrorMessage")

    response: Dict[str, Any] = Field(..., alias="Response")


class GPTServiceError(Exception):
    """Raised when the service returns an error response or the response times out."""


class GPTServiceConnector:
    """Client class for using GPT Service from outside.

    Communicates with the service over Redis. Every request is pushed onto the
    `request_stream` queue; the service processes it and writes the response to
    the request-specific `llm:response:{request_id}` queue. The `send` call
    blocks until that response arrives (or the timeout is reached).

    Important: the `db`, `request_stream` and `response_stream` values here must
    exactly match those on the service side (Messenger); otherwise requests and
    responses will not line up.

    Example usage:
        async with GPTServiceConnector("localhost", 6379) as gpt:
            res = await gpt.send("Hello", system_prompt="Return JSON only.")
            print(res.response)
    """

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        request_stream: str = "llm:request",
        response_stream: str = "llm:response:{request_id}",
        timeout: float = 600.0,
    ):
        """
        :param redis_host: Redis server host.
        :param redis_port: Redis server port.
        :param request_stream: Name of the queue requests are pushed to.
        :param response_stream: Template of the queue responses are read from;
            `{request_id}` is replaced with the matching id on each request.
        :param timeout: Maximum number of seconds to wait for a response. LLM
            calls may be slow, so the default is 600 s (10 min).
        """
        self.redis_client = aioredis.Redis(
            host=redis_host, port=redis_port, db=1, decode_responses=True
        )
        self.request_stream = request_stream
        self.response_stream = response_stream
        self.timeout = timeout

    async def send(
        self,
        user_prompt: str,
        model: str = 'gpt-5.4-mini',
        system_prompt: Optional[str] = None,
        table_code: str = 'llm',
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ResponseModel:
        """Sends a request to the service and waits for the response.

        :param user_prompt: User message to pass to the model.
        :param model: Name of the model to use.
        :param system_prompt: Optional system message; omitted if not provided.
        :param table_code: Prefix of the request id (`request_id`). Must be
            compatible with the validation on the service side.
        :param reasoning_effort: Optional reasoning effort (`none`, `minimal`,
            `low`, `medium`, `high`, `xhigh`); the service default if omitted.
        :param temperature: Optional sampling temperature (0-2); the service
            default if omitted. Dropped by the service when the model rejects it.
        :raises GPTServiceError: On timeout or when the service returns an error.
        :return: The ``ResponseModel`` returned by the service.
        """
        request_id = f"{table_code}-{uuid.uuid4()}"

        request = {
            "Model": model,
            "RequestId": request_id,
            "UserPrompt": user_prompt,
        }

        if system_prompt is not None:
            request["SystemPrompt"] = system_prompt

        if reasoning_effort is not None:
            request["ReasoningEffort"] = reasoning_effort

        if temperature is not None:
            request["Temperature"] = temperature

        # Push the request onto the queue the service listens on.
        await self.redis_client.rpush(self.request_stream, json.dumps(request))

        await anyio.sleep(0.1)

        # Resolve the request-specific response queue and wait for the response.
        response_address = self.response_stream.replace("{request_id}", request_id)

        res = await self.redis_client.blpop(response_address, timeout=self.timeout)

        if res is None:
            raise GPTServiceError(
                f"No response for request `{request_id}` within {self.timeout} s."
            )

        _, item = res

        response = ResponseModel.model_validate(json.loads(item))

        # Raise instead of silently passing on a service-side error.
        if response.status == 'Error':
            raise GPTServiceError(
                f"Service error ({request_id}): {response.error_message}"
            )

        return response

    async def close(self):
        """Closes the Redis connection. Called automatically when used as an
        `async with` context manager."""
        await self.redis_client.aclose()

    async def __aenter__(self) -> "GPTServiceConnector":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()