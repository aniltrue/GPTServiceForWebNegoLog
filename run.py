import anyio

from Service import Service


async def main():
    service = Service()

    try:
        await service.start_service()
    finally:
        # Stop the loop and release the messenger/logger on shutdown.
        service.working = False
        await service.messenger.close()
        await service.logger.close()


if __name__ == "__main__":
    try:
        anyio.run(main)
    except KeyboardInterrupt:
        print("GPT Service stopped.")
