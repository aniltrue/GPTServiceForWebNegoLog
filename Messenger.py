import asyncio
import json
import os
from typing import Optional, Dict, Any

import redis
import redis.asyncio as aioredis
from dotenv import load_dotenv

from Models import RequestModel, ResponseModel

load_dotenv(dotenv_path='service.env')

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REQUEST_STREAM = os.getenv('REQUEST_STREAM', "llm:request")
RESPONSE_STREAM = os.getenv('RESPONSE_STREAM', "llm:response:{request_id}")


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

    async def send_response(self, request_id: str, response: Dict[str, Any]):
        response_address = RESPONSE_STREAM.replace("{request_id}", request_id)

        await self.redis_client.rpush(response_address, ResponseModel(
            Status='Success',
            RequestId=request_id,
            Response=response
        ).model_dump_json(by_alias=True))

    async def send_error(self, request_id: str, error_message: Optional[str]):
        response_address = RESPONSE_STREAM.replace("{request_id}", request_id)

        await self.redis_client.rpush(response_address, ResponseModel(
            Status='Error',
            RequestId=request_id,
            Response={},
            ErrorMessage=error_message
        ).model_dump_json(by_alias=True))

    async def start(self):
        self.redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=True)

    async def close(self):
        if self.redis_client is None:
            return

        await self.redis_client.connection_pool.disconnect()
        await self.redis_client.close()
