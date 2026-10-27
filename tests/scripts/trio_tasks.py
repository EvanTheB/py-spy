import time

import trio


def burn():
    end = time.perf_counter() + 0.05
    while time.perf_counter() < end:
        pass


async def busy():
    while True:
        burn()
        await trio.sleep(0)


async def sleeper():
    await trio.sleep(10000)


async def parent():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(sleeper, name="sleeper")


async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(busy, name="busy")
        nursery.start_soon(parent, name="parent")


if __name__ == "__main__":
    trio.run(main)
