import os
import sys
import asyncio
import random

import discord
import httpx
from dotenv import load_dotenv

load_dotenv()


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
            ms = await message.reply("\u200b:clock3:", mention_author=False)
            async with message.channel.typing():
                prompt = message.content.replace(self.user.mention, "\u200b").strip()

                async with httpx.AsyncClient(
                    timeout=None
                ) as client:  # queue a response from the API
                    response = await client.post(
                        "http://127.0.0.1:8001/gpt", json={"prompt": prompt}
                    )
                    if response.status_code == 200:
                        gpt_message = response.json()["message"]
                        if len(gpt_message) <= 2000:
                            return await ms.edit(
                                gpt_message
                            )  # everything worked if the code got here
                        else:
                            await ms.edit("\u200b:scroll::warning:")
                            await asyncio.sleep(30)
                            return await ms.delete()

                    else:
                        user = self.get_user(603635602809946113)  # replace with ID of the owner's ID you want to send errors to
                        newline = "\n"
                        await ms.edit("\u200b:scroll::x:")
                        await user.send(
                            f"# Oh shit!\n"
                            f"Error {response.json()['code']} has occurred: {response.json()['error']}\n"
                            f"The following errors were caught:\n{newline.join(response.json()['errors'])}\n\n"
                            f"If someone else got this error, tell them to retry their request."
                        )  # comprehensive error reports, only the owner will get those
                        await asyncio.sleep(30)
                        return await ms.delete()


client = Clyde(max_messages=None, chunk_guilds_at_startup=False)  # this speeds-up selfbot boots
client.run(os.getenv("TOKEN"))  # be careful with posting a token here
