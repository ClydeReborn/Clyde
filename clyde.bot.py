import os

print(f"Your virtual environment path: {os.getenv('VIRTUAL_ENV')}")

import asyncio
import secrets
import random

import discord
from discord.ext import commands

import httpx
from dotenv import load_dotenv

# prevent leaking any tokens to cause bans or blocks
load_dotenv()

owner = os.getenv("OWNER")
error_channel = os.getenv("ERROR_CHANNEL")
intents = discord.Intents.all()
bot_name = "Clyde"

client = commands.Bot(
    command_prefix=secrets.token_urlsafe(8),
    help_command=None,
    intents=intents,
    owner_id=owner,
)


@client.event
async def on_connect():
    print(f"{bot_name} has connected to Discord.")


@client.event
async def on_ready():
    print(f"{bot_name} ready!")


@client.event
async def on_guild_join(guild):
    me = guild.get_member(client.user.id)
    await me.edit(nick=bot_name)


@client.event
async def on_message(message):
    # specify user ids forbidden from using Clyde, eg. 691187099570929684
    forbidden = []

    if message.author.id in forbidden:
        print("Ignoring message")
        return

    if message.author == client.user:
        print("Ignoring self message")
        return

    if message.author.bot:
        print("Ignoring bot message")
        return

    if (
        client.user.mentioned_in(message)
        or not message.guild
        or message.channel.type == discord.ChannelType.public_thread
    ):
        clyde_error_messages = [
            "Oops, I seem to have hit a snag! My creators are on the case, and I'll be back to normal soon.",
            "Woah there, I've hit a bump in the road. My creators have been notified and I'll be fixed shortly.",
            "Sorry, something went wrong. My creators have been notified and I'll be fixed shortly.",
            "Oops, I've run into a problem. But don't worry, my team is on it and I'll be back to full strength soon.",
            "Uh oh, I've encountered an issue. Rest assured, I have my best people working on the problem.",
        ]
        async with message.channel.typing():
            prompt = message.content.replace(client.user.mention, "\u200b").strip()

            async with httpx.AsyncClient(timeout=None) as web:
                try:
                    # run ai externally via an attached api
                    response = await web.post(
                        "http://127.0.0.1:8001/gpt",
                        json={"prompt": prompt, "type": "g4f"},
                    )
                except httpx.ConnectError:
                    # server offline error response
                    ms = await message.reply(
                        random.choice(clyde_error_messages), mention_author=False
                    )
                    channel = await client.fetch_channel(error_channel)
                    await channel.send(
                        "# @everyone, we have an error!\nError 2 has occurred: The API server is offline.\n\n"
                        "Please restart the API server before trying to use ChatGPT."
                    )
                    await asyncio.sleep(30)
                    return await ms.delete()

                if response.status_code == 200:
                    # correct response
                    gpt_message = response.json()["message"]
                    if 0 < len(gpt_message) <= 2000:
                        return await message.reply(gpt_message)

            async with httpx.AsyncClient(timeout=None) as web:
                try:
                    # retry the above if failed with g4f
                    response = await web.post(
                        "http://127.0.0.1:8001/gpt",
                        json={"prompt": prompt, "type": "tgpt"},
                    )
                except httpx.ConnectError:
                    # server offline error response
                    ms = await message.reply(
                        random.choice(clyde_error_messages), mention_author=False
                    )
                    channel = await client.fetch_channel(error_channel)
                    await channel.send(
                        "# @everyone, we have an error!\nError 2 has occurred: The API server is offline.\n\n"
                        "Please restart the API server before trying to use ChatGPT."
                    )
                    await asyncio.sleep(30)
                    return await ms.delete()

                if response.status_code == 200:
                    # correct response
                    gpt_message = response.json()["message"]
                    if 0 < len(gpt_message) <= 2000:
                        return await message.reply(gpt_message)

                # error response
                channel = await client.fetch_channel(error_channel)
                newline = "\n"
                ms = await message.channel.send(
                    random.choice(clyde_error_messages), mention_author=False
                )
                await channel.send(
                    f"# @everyone, we have an error!\n"
                    f"Error {response.json()['code']} has occurred: {response.json()['error']}\n"
                    f"The recent errors caught were:\n{newline.join(response.json()['errors'][-5:])}\n\n"
                    f"If someone else got this error, tell them to retry their request."
                )
                await asyncio.sleep(5)
                return await ms.delete()


client.run(os.getenv("TOKEN"))
