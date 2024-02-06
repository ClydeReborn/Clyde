import os
import sys
import random

import discord
import httpx


class Clyde(discord.Client):
    async def on_connect(self):
        print("Clyde has connected to Discord.")

    async def on_ready(self):
        print("Clyde ready!")

    async def on_message(self, message):  # ignore if:
        forbidden = []  # block user ids from accessing Clyde here

        if message.author.id in forbidden:  # from blocked user
            print("Ignoring message")
            return

        if message.author == self.user:  # received message from itself
            print("Ignoring self message")
            return

        if message.author.bot:  # received message from a bot
            print("Ignoring bot message")
            return

        if (  # process if:
            self.user.mentioned_in(message)  # pinged
            or not message.guild  # received message in a dm
            or message.channel.type
            == discord.ChannelType.public_thread  # received message in a thread
        ):
            async with message.channel.typing():
                prompt = message.content.replace(self.user.mention, "").strip()

                async with httpx.AsyncClient(
                    timeout=None
                ) as client:  # queue a response from the API
                    response = await client.post(
                        "http://127.0.0.1:8001/gpt", json={"prompt": prompt}
                    )
                    if response.status_code == 200:
                        gpt_message = response.json()["message"]
                        await message.reply(
                            gpt_message, mention_author=False
                        )  # everything worked if the code got here
                    else:
                        newline = "\n"
                        error_messages = [
                            "Oops, I seem to have hit a snag! My creators are on the case, and I'll be back to normal soon.",
                            "Woah there, I've hit a bump in the road. My creators have been notified and I'll be fixed shortly.",
                            "Sorry, something went wrong. My creators have been notified and I'll be fixed shortly.",
                            "Oops, I've run into a problem. But don't worry, my team is on it and I'll be back to full strength soon.",
                            "Uh oh, I've encountered an issue. Rest assured, I have my best people working on the problem.",
                        ]
                        return await message.reply(
                            f"{random.choice(error_messages)}\n\n"
                            f"Code: {response.json()['code']}, {response.json()['error']}\n"
                            f"Errors returned:\n{newline.join(response.json()['errors'])}\n\n"
                            f"This may be a one-time issue, try again!",
                            mention_author=False,
                        )  # comprehensive error report with Clyde's original error message


client = Clyde(max_messages=None, chunk_guilds_at_startup=False)
client.run(
    "<TOKEN_GOES_HERE>"
)  # be careful with posting a token here
