import argparse

import anyio

from Service import Service


async def main(clean: bool):
    service = Service()

    try:
        await service.start_service(clean=clean)
    finally:
        # Stop the loop and release the messenger/logger on shutdown.
        service.working = False
        await service.messenger.close()
        await service.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Service")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the waiting requests and the unread responses in Redis before starting")
    args = parser.parse_args()

    try:
        anyio.run(main, args.clean)
    except KeyboardInterrupt:
        print("GPT Service stopped.")
