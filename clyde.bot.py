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
intents = discord.Intents.default()
intents.message_content = True

errors = [
    "Oops, I seem to have hit a snag! My creators are on the case, and I'll be back to normal soon.",
    "Woah there, I've hit a bump in the road. My creators have been notified and I'll be fixed shortly.",
    "Sorry, something went wrong. My creators have been notified and I'll be fixed shortly.",
    "Oops, I've run into a problem. But don't worry, my team is on it and I'll be back to full strength soon.",
    "Uh oh, I've encountered an issue. Rest assured, I have my best people working on the problem.",
]  # Original error messages from Clyde, late November


class Clyde(commands.Bot):
    """A slash-command enabled bot class."""
    async def setup_hook(self):
        synced = await self.tree.sync()
        print(f"Synced with data: {len(synced)} commands")


class AI(commands.Cog):
    """A feature class containing the main AI functions."""
    def __init__(self, client):
        self.client = client

    @discord.app_commands.command()
    @discord.app_commands.describe(prompt="The prompt to query Clyde with.", image="Image to use with the prompt.")
    @discord.app_commands.allowed_installs(guilds=True, users=True)
    @discord.app_commands.allowed_contexts(dms=True, guilds=True, private_channels=True)
    async def ai(self, interaction: discord.Interaction, prompt: str, image: discord.Attachment = None):
        """Fetch an AI response."""
        await interaction.response.defer(thinking=True)
        async with httpx.AsyncClient(timeout=None) as web:
            try:
                # run ai externally via an attached api
                response = await web.post(
                    "http://127.0.0.1:8001/gpt",
                    json={"prompt": prompt, "image": image.url if image else None, "type": "gemini"},
                )
            except httpx.ConnectError as reason:
                ms = await interaction.followup.send(random.choice(errors))
                await asyncio.sleep(10)
                await ms.delete()
                raise RuntimeError("API is offline") from reason

            if response.status_code == 200:
                gpt_message = response.json()["message"]
                if 0 <= len(gpt_message) <= 2000:
                    return await interaction.followup.send(gpt_message)
                else:
                    return await interaction.followup.send("Woah, that's a big one. Try sending less context.")

            ms = await interaction.followup.send(random.choice(errors))
            await asyncio.sleep(10)
            await ms.delete()
            raise RuntimeError("API didn't work")

    # @commands.Cog.listener()
    # async def on_guild_join(self, guild):
    #    """Rename to 'Clyde' after server join, adds green @tag in mobile 
    #    me = guild.get_member(self.client.user.id)
    #    await me.edit(nick="Clyde")

    @commands.Cog.listener()
    async def on_connect(self):
        """Notify when connected."""
        print("Clyde has connected to Discord.")

    @commands.Cog.listener()
    async def on_ready(self):
        """Notify when ready."""
        print("Clyde ready!")

    @commands.Cog.listener()
    async def on_message(self, message):
        """Query AI messages without using slash commands for servers that have Clyde in the server."""
        # Ban users from using Clyde, eg. 691187099570929684
        forbidden = []

        if message.author.id in forbidden:
            print("Ignoring message")
            return

        if message.author == self.client.user:
            print("Ignoring self message")
            return

        if message.author.bot:
            print("Ignoring bot message")
            return

        if (
            self.client.user.mentioned_in(message)
            or not message.guild
            or message.channel.type == discord.ChannelType.public_thread
        ):
            async with message.channel.typing():
                prompt = message.content.replace(
                    self.client.user.mention, "Clyde"
                ).strip()

                async with httpx.AsyncClient(timeout=None) as web:
                    try:
                        # run ai externally via an attached api
                        response = await web.post(
                            "http://127.0.0.1:8001/gpt",
                            json={"prompt": prompt, "type": "gemini"},
                        )
                    except httpx.ConnectError as reason:
                        # server offline error response
                        ms = await message.reply(
                            random.choice(errors), mention_author=False
                        )
                        await asyncio.sleep(10)
                        await ms.delete()
                        raise RuntimeError("API is offline") from reason

                    if response.status_code == 200:
                        # correct response
                        gpt_message = response.json()["message"]
                        if 0 < len(gpt_message) <= 2000:
                            return await message.reply(gpt_message)
                        else:
                            return await message.reply("Woah, that's a big one. Try sending less context.")

                    # error response
                    ms = await message.reply(
                        random.choice(errors), mention_author=False
                    )
                    await asyncio.sleep(10)
                    await ms.delete()
                    raise RuntimeError("API didn't work")


async def main():
    """Initialization function ran by asyncio."""
    client = Clyde(
        command_prefix=secrets.token_urlsafe(8),  # A prefix is required, but Clyde is supposed to have None
        help_command=None,
        intents=intents,
        owner_id=owner,
    )
    await client.add_cog(AI(client))
    await client.start(os.getenv("TOKEN"))


asyncio.run(main())
