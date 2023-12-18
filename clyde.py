import os
import sys
import random

import discord
import httpx


class Clyde(discord.Client):
    async def on_connect(self):
        print("Clyde has connected to Discord.")

    async def on_disconnect(self):
        print("Connection failure, rebooting Clyde...")
        os.execl(sys.executable, sys.executable, *sys.argv)
        sys.exit(0)

    async def on_ready(self):
        print("Clyde ready!")

    async def on_message(self, message):
        forbidden = []

        if message.author.id in forbidden:
            print("Ignoring message")
            return

        if message.author == self.user:
            print("Ignoring self message")
            return

        if message.author.bot:
            print("Ignoring bot message")
            return

        if (
            self.user.mentioned_in(message)
            or not message.guild
            or message.channel.type == discord.ChannelType.public_thread
        ):
            async with message.channel.typing():
                prompt = message.content.replace(self.user.mention, "Sakoma").strip()
                errors = 0

                while True:
                    async with httpx.AsyncClient(timeout=None) as client:
                        response = await client.post(
                            "http://127.0.0.1:8001/gpt", json={"prompt": prompt}
                        )
                        if response.status_code == 200:
                            gpt_message = response.json()["message"]
                            await message.reply(gpt_message, mention_author=False)
                            break
                        else:
                            if errors < 5:
                                continue
                            else:
                                error_messages = [
                                    "Oops, I seem to have hit a snag! My creators are on the case, and I'll be back to normal soon.",
                                    "Woah there, I've hit a bump in the road. My creators have been notified and I'll be fixed shortly.",
                                    "Sorry, something went wrong. My creators have been notified and I'll be fixed shortly.",
                                    "Oops, I've run into a problem. But don't worry, my team is on it and I'll be back to full strength soon.",
                                    "Uh oh, I've encountered an issue. Rest assured, I have my best people working on the problem.",
                                ]
                                return await message.reply(
                                    random.choice(error_messages),
                                    mention_author=False,
                                    delete_after=5,
                                )


client = Clyde(max_messages=None, chunk_guilds_at_startup=False)
client.run("TOKEN_GOES_HERE")
