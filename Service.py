import os

import anyio
from anyio import WouldBlock
from dotenv import load_dotenv

from AsyncLogger import get_logger
from LLMProcess import LLMProcessor
from Messenger import Messenger
from Models import RequestModel

load_dotenv(dotenv_path='service.env')
load_dotenv(dotenv_path='.env')

MAX_RETRY = int(os.getenv("MAX_RETRY", 10))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 10))


class Service:
    def __init__(self):
        self.logger = get_logger("GPTService")
        self.messenger = Messenger()
        self.processor = LLMProcessor(self.logger)

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
                    request = None
                    request = await self.messenger.listen_request()

                    if request is None:
                        self.semaphore.release()
                        await anyio.sleep(0.01)  # backoff
                        continue

                    tg.start_soon(self.process, request, name=request.request_id)
                except Exception as e:
                    await self.logger.error(f"{e}")

                    if request is not None:
                        await self.messenger.send_error(request.request_id, str(e))

                    self.semaphore.release()

    async def process(self, request: RequestModel):
        try:
            response = await self.processor.run(request)

            await self.messenger.send_response(request.request_id, response)
        except Exception as e:
            await self.messenger.send_error(request.request_id, str(e))

            await self.logger.error(f"Request ({request.request_id}): {e}")
        finally:
            self.semaphore.release()
