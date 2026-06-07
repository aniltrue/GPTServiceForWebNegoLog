import os
from typing import Dict

import anyio
from anyio import WouldBlock
from dotenv import load_dotenv

from AsyncLogger import AsyncLogger
from Messenger import Messenger
from Models import RequestModel

load_dotenv(dotenv_path='service.env')
load_dotenv(dotenv_path='.env')

MAX_RETRY = int(os.getenv("MAX_RETRY", 10))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 10))


class Service:
    def __init__(self):
        self.logger = AsyncLogger("GPTService")
        self.messenger = Messenger()

        self.semaphore = anyio.Semaphore(MAX_CONCURRENCY)

        self.working = False

    async def start_service(self):
        self.working = True

        await self.logger.start()
        await self.messenger.start()
        await self.logger.info("GPT Service Started.")

        async with anyio.create_task_group() as tg:
            while self.working:
                try:
                    self.semaphore.acquire_nowait()
                except WouldBlock:
                    await anyio.sleep(0.01)
                    continue

                try:
                    request = await self.messenger.listen_request()

                    if request is None:
                        self._sema.release()
                        await anyio.sleep(0.01)  # backoff
                        continue

                    tg.start_soon(self.process, request, name=request.request_id)

                    self.semaphore.release()
                except Exception as e:
                    await self.logger.error(f"{e}")
                    self.semaphore.release()

    async def process(self, request: RequestModel):
        ...