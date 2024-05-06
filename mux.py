import asyncio
import sys


async def _read_stream(stream, cb):
    while True:
        line = await stream.readline()
        if line:
            cb(line)
        else:
            break


async def _stream_subprocess(cmd, stdout_cb, stderr_cb):
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    await asyncio.gather(
        _read_stream(process.stdout, stdout_cb), _read_stream(process.stderr, stderr_cb)
    )
    return await process.wait()


def execute(cmd, stdout_cb, stderr_cb):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    rc = loop.run_until_complete(
        _stream_subprocess(
            cmd,
            stdout_cb,
            stderr_cb,
        )
    )
    loop.close()
    return rc


if __name__ == "__main__":
    exit_code = execute(
        ["bash", "-c", "python -u api.py & python -u clyde.py"],
        lambda out: print(out.decode("utf-8", errors="ignore"), end=""),
        lambda err: print(err.decode("utf-8", errors="ignore"), end=""),
    )
    sys.exit(exit_code)
