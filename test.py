import os

import anyio
from dotenv import load_dotenv

from GPTConnector import GPTServiceConnector, GPTServiceError

load_dotenv(dotenv_path="service.env")
load_dotenv(dotenv_path=".env")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


async def main():
    async with GPTServiceConnector(REDIS_HOST, REDIS_PORT) as gpt:
        try:
            res = await gpt.send(
                user_prompt="Return a JSON object with a single field `greeting` "
                            "set to a short hello message.",
                system_prompt="You are a helpful assistant. Respond ONLY with valid JSON.",
            )
        except GPTServiceError as e:
            print(f"[FAIL] {e}")
            return

        print(f"[OK] Status : {res.status}")
        print(f"     Response: {res.response}")


if __name__ == "__main__":
    anyio.run(main)
