import asyncio
import json
import os
from typing import Optional, Dict, Any, Tuple

import redis
import redis.asyncio as aioredis
from dotenv import load_dotenv

from Models import RequestModel, ResponseModel

load_dotenv(dotenv_path='service.env')

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REQUEST_STREAM = os.getenv('REQUEST_STREAM', "llm:request")
RESPONSE_STREAM = os.getenv('RESPONSE_STREAM', "llm:response:{request_id}")
# A response nobody reads (the client timed out) is dropped after this many seconds
RESPONSE_TTL = int(os.getenv('RESPONSE_TTL', 3600))


class Messenger:
    def __init__(self, time_out: float = 5):
        self.redis_client: Optional[aioredis.Redis] = None
        self.time_out: float = time_out

    async def listen_request(self) -> Optional[RequestModel]:
        try:
            res = await self.redis_client.blpop(REQUEST_STREAM, timeout=self.time_out)
        except redis.asyncio.TimeoutError:
            return None

        if res is None:
            return None

        _, item = res

        return RequestModel.model_validate(json.loads(item))

    async def _push_response(self, request_id: str, response: ResponseModel):
        response_address = RESPONSE_STREAM.replace("{request_id}", request_id)

        async with self.redis_client.pipeline(transaction=True) as pipe:
            pipe.rpush(response_address, response.model_dump_json(by_alias=True))
            pipe.expire(response_address, RESPONSE_TTL)
            await pipe.execute()

    async def send_response(self, request_id: str, response: Dict[str, Any]):
        await self._push_response(request_id, ResponseModel(
            Status='Success',
            RequestId=request_id,
            Response=response
        ))

    async def send_error(self, request_id: str, error_message: Optional[str]):
        await self._push_response(request_id, ResponseModel(
            Status='Error',
            RequestId=request_id,
            Response={},
            ErrorMessage=error_message
        ))

    async def clean(self) -> Tuple[int, int]:
        """
            Deletes the waiting requests and the unread responses. Returns their numbers.
        """
        requests = await self.redis_client.llen(REQUEST_STREAM)

        response_keys = [key async for key in
                         self.redis_client.scan_iter(match=RESPONSE_STREAM.replace("{request_id}", "*"))]

        if requests > 0 or response_keys:
            await self.redis_client.delete(REQUEST_STREAM, *response_keys)

        return requests, len(response_keys)

    async def start(self):
        self.redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=True)

    async def close(self):
        if self.redis_client is None:
            return

        await self.redis_client.connection_pool.disconnect()
        await self.redis_client.close()
