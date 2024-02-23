import subprocess
import asyncio
import sys

api = subprocess.Popen(
    ["python", "api.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE
)
clyde = subprocess.Popen(
    ["python", "clyde.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
)


async def output(proc, buf):
    for c in iter(lambda: proc.read(1), b""):
        buf.write(c)


async def main():
    tasks = [
        output(api.stderr, sys.stderr.buffer),
        output(clyde.stderr, sys.stderr.buffer),
        output(clyde.stdout, sys.stdout.buffer),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
