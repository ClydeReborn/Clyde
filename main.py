import os
import io
import base64
import difflib
import logging
import asyncio
import contextlib
from typing import Optional, Tuple, Dict, List

import hikari
import lightbulb
import miru
import httpx
from google import genai
from google.genai import types
from groq import AsyncGroq
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_discord_bot")

# Load environment variables
load_dotenv()

# Constants
MAX_DISCORD_MESSAGE_LENGTH = 2000
MAX_CHAT_HISTORY = 30
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_PROMPT = "You are a helpful assistant."

# Model lists for autocomplete
MODELS = {
    "gemini": [
        # Gemini 2.5 series
        "gemini-2.5-pro-preview-03-25",
        # Gemini 2.0 series
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-thinking-exp-01-21",
        # Gemini 1.5 series
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ],
    "gemma": [
        # Gemma 2 series
        "gemma-2-2b-it",
        "gemma-2-27b-it",
        # Gemma 3 series
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
    ],
    "other": [
        # Qwen models
        "qwen-qwq-32b",
        # DeepSeek models
        "deepseek-r1-distill-llama-70b",
        # Llama 3.x series
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        # Llama 4 series
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-7b-16e-instruct",
        # Mistral models
        "mistral-saba-24b",
    ],
}

# Prompt presets
PROMPT_PRESETS = {
    "default": "You are a helpful assistant.",
    "gpt": "You are ChatGPT, a large language model trained by OpenAI.",
    "casual": "You are a casual assistant.",
    "discord": "You are a Discord user, respond casually and lowercase only.",
    "programmer": "You are a programming assistant who provides concise and clean code examples.",
    "storyteller": "You are a creative storyteller who can craft engaging narratives.",
}

# Initialize bot
bot = hikari.GatewayBot(
    token=os.getenv("DISCORD_BOT_TOKEN"),
    intents=hikari.Intents.GUILD_MESSAGES | hikari.Intents.MESSAGE_CONTENT,
)
inter_client = miru.Client(bot)
client = lightbulb.client_from_app(bot)
ai_group = lightbulb.Group("ai", "AI command group")

# Initialize API clients
try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_TOKEN"))
    groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_TOKEN"))
except Exception as e:
    logger.error(f"Failed to initialize API clients: {e}")
    raise

# Store user chat histories
chat_histories: Dict[int, List[str]] = {}


class AIView(miru.View):
    def __init__(self, entries: list[str]) -> None:
        super().__init__(timeout=60)  # Increased timeout to 60 seconds is more user-friendly
        self.entries = entries
        self.index = 0

        # Initialize buttons with proper disabled states
        self.previous_page.disabled = True
        self.next_page.disabled = len(entries) <= 1

    @miru.button(emoji="◀️", style=hikari.ButtonStyle.SECONDARY)
    async def previous_page(self, ctx: miru.ViewContext, button: miru.Button) -> None:
        self.index -= 1
        self._update_buttons()
        await ctx.edit_response(self.entries[self.index], components=self.build())

    @miru.button(label="▶️", style=hikari.ButtonStyle.SECONDARY)
    async def next_page(self, ctx: miru.ViewContext, button: miru.Button) -> None:
        self.index += 1
        self._update_buttons()
        await ctx.edit_response(self.entries[self.index], components=self.build())

    def _update_buttons(self) -> None:
        """Helper method to update button states."""
        self.previous_page.disabled = self.index == 0
        self.next_page.disabled = self.index == len(self.entries) - 1

    async def view_check(self, ctx: miru.ViewContext) -> bool:
        """Check if the user interacting is the one who invoked the command."""
        return ctx.user.id == ctx.author.id  # Simplified return

    async def on_timeout(self) -> None:
        """Disable all buttons when the view times out."""
        for child in self.children:
            child.disabled = True

        with contextlib.suppress(hikari.ForbiddenError):
            await self.message.edit(components=self.build())


class AIService:
    """Service for handling AI-related operations"""

    @staticmethod
    def format_chat_history(messages: List[str]) -> str:
        """Format chat history for AI consumption"""
        formatted = ""
        for i, message in enumerate(messages[:MAX_CHAT_HISTORY], start=1):
            role = "User" if i % 2 == 0 else "Assistant"
            formatted += f"{role}: {message}\n"
        return formatted

    @staticmethod
    async def generate_text_with_gemini(
            request: str,
            model: str,
            system_prompt: str,
            user_id: int,
            image: Optional[io.BytesIO] = None,
    ) -> Tuple[str, int]:
        """Generate text using Gemini models"""
        try:
            config = types.GenerateContentConfig(system_instruction=system_prompt)

            if image:
                payload = [
                    request,
                    types.Part.from_bytes(
                        data=image.read(),
                        mime_type="image/png",
                    ),
                ]
            else:
                chat_histories.setdefault(user_id, []).append(request)
                formatted_history = AIService.format_chat_history(chat_histories[user_id])
                payload = [formatted_history]

            # Choose the correct API based on model
            if "gemma" in model.lower():
                response = gemini_client.models.generate_content(model=model, contents=payload)
            else:
                response = gemini_client.models.generate_content(
                    model=model, config=config, contents=payload
                )

            # Sometimes the roles are shown to the user, but they are supposed to be an internal guideline on how the AI chat is going on
            result = response.text.replace("Assistant: ", "").replace("User: ", "")

            chat_histories[user_id].append(result)
            return result, 200
        except Exception as exc:
            logger.error(f"Gemini generation error: {exc}")
            return str(e), 500

    @staticmethod
    async def generate_text_with_groq(
            request: str, model: str, system_prompt: str, user_id: int
    ) -> Tuple[str, int]:
        """Generate text using Groq models"""
        try:
            chat_histories.setdefault(user_id, []).append(request)
            formatted_history = AIService.format_chat_history(chat_histories[user_id])

            response = await groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_history},
                ],
            )

            result = response.choices[0].message.content
            result = result.replace("Assistant: ", "")

            chat_histories[user_id].append(result)
            return result, 200
        except Exception as exc:
            logger.error(f"Groq generation error: {exc}")
            return str(e), 500

    @staticmethod
    async def generate_image(prompt: str) -> Tuple[Optional[io.BytesIO | str], int]:
        """Generate an image from text prompt"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as img_client:
                response = await img_client.post(
                    "https://ir-api.myqa.cc/v1/openai/images/generations",
                    json={
                        "model": "google/gemini-2.0-flash-exp:free",
                        "prompt": prompt,
                    },
                    headers={
                        "Authorization": f"Bearer {os.getenv('IMAGE_GEN_TOKEN')}",
                        "Content-Type": "application/json",
                    },
                )

                response.raise_for_status()
                data = response.json()

                if "data" not in data or not data["data"]:
                    return "Empty response from image service", 500

                image_b64 = data["data"][0]["b64_json"]
                image = io.BytesIO(base64.b64decode(image_b64))
                image.seek(0)

                return image, 200

        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error during image generation: {exc.response.status_code} - {exc.response.text}")
            return f"HTTP error: {exc.response.status_code}", exc.response.status_code
        except Exception as exc:
            logger.error(f"Image generation error: {exc}")
            return str(e), 500


async def generate_text(
        request: str,
        model: str,
        prompt: str,
        user_id: int,
        image: Optional[io.BytesIO] = None,
) -> Tuple[str, int]:
    """Unified text generation function that routes to appropriate service"""
    try:
        # Route to appropriate service based on model name
        if any(name in model.lower() for name in ["gemma", "gemini"]):
            return await AIService.generate_text_with_gemini(
                request, model, prompt, user_id, image
            )
        if image:
            return "Image processing not supported for non-Gemini models", 400
        return await AIService.generate_text_with_groq(request, model, prompt, user_id)
    except Exception as exc:
        logger.error(f"Text generation error: {exc}")
        return f"An unexpected error occurred: {str(exc)}", 500


# Event handlers
async def on_starting(_: hikari.StartingEvent) -> None:
    """Handle bot startup event"""
    await client.start()
    logger.info("Bot is starting")
bot.subscribe(hikari.StartingEvent, on_starting)


async def on_started(_: hikari.StartedEvent) -> None:
    """Handle bot started event"""
    logger.info("Bot is now running")
bot.subscribe(hikari.StartedEvent, on_started)


# Autocomplete handlers
async def model_autocomplete(ctx: lightbulb.AutocompleteContext) -> None:
    """Provide model autocomplete suggestions"""
    current_value: str = ctx.focused.value or ""

    # Flatten the model list
    all_models = [model for category in MODELS.values() for model in category]

    # Direct matches first
    prefix_matches = [m for m in all_models if m.lower().startswith(current_value.lower())]
    if prefix_matches:
        return await ctx.respond(prefix_matches[:25])

    # Then fuzzy matches
    fuzzy_matches = difflib.get_close_matches(current_value, all_models, n=10, cutoff=0.3)
    if fuzzy_matches:
        return await ctx.respond(fuzzy_matches)

    # Default suggestions grouped by type
    suggestions = []
    for category, models in MODELS.items():
        suggestions.extend(models[:3])  # Take first 3 from each category

    await ctx.respond(suggestions[:25])


async def prompt_preset_autocomplete(ctx: lightbulb.AutocompleteContext) -> None:
    """Provide prompt preset autocomplete suggestions"""
    current_value: str = ctx.focused.value or ""

    # Direct matches first
    prefix_matches = [k for k in PROMPT_PRESETS if k.lower().startswith(current_value.lower())]
    if prefix_matches:
        return await ctx.respond(prefix_matches)

    # Then fuzzy matches
    fuzzy_matches = difflib.get_close_matches(current_value, PROMPT_PRESETS.keys(), n=5, cutoff=0.3)
    if fuzzy_matches:
        return await ctx.respond(fuzzy_matches)

    # Default is all presets
    await ctx.respond(list(PROMPT_PRESETS.keys()))


# Command handlers
@ai_group.register()
class AIText(lightbulb.SlashCommand, name="text", description="Generate text with AI"):
    request: str = lightbulb.string("request", "The request to send to the AI.")
    model: Optional[str] = lightbulb.string(
        "model",
        "The model to use.",
        default=DEFAULT_MODEL,
        autocomplete=model_autocomplete,
    )
    prompt: Optional[str] = lightbulb.string(
        "prompt",
        "The prompt or preset to use.",
        default="default",
        autocomplete=prompt_preset_autocomplete,
    )

    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        await ctx.defer(ephemeral=False)

        # Resolve prompt preset if needed
        resolved_prompt = PROMPT_PRESETS.get(self.prompt, self.prompt)

        try:
            response, status = await generate_text(
                self.request, self.model, resolved_prompt, ctx.interaction.user.id
            )

            if status == 200:
                # Handle long responses
                if len(response) > MAX_DISCORD_MESSAGE_LENGTH:
                    chunks = []
                    while response:
                        split_idx = response.rfind("\n\n", 0, MAX_DISCORD_MESSAGE_LENGTH)

                        if split_idx in [-1, 0]:
                            split_idx = MAX_DISCORD_MESSAGE_LENGTH

                        chunk = response[:split_idx].rstrip()
                        response = response[split_idx:].lstrip()
                        chunks.append(chunk)
                    view = AIView(chunks)
                    await ctx.respond(chunks[0], components=view)
                    inter_client.start_view(view)
                else:
                    await ctx.respond(response)
            else:
                embed = hikari.Embed(
                    title="Error",
                    description=f"An error occurred:\n```py\n{str(response)}\n```",
                    color=hikari.Color.from_hex_code("#ed4245"),
                )
                embed.set_footer(f"Status code {status}")

                await ctx.respond(
                    embed=embed,
                    flags=hikari.MessageFlag.EPHEMERAL,
                )
        except Exception as exc:
            logger.error(f"Error in AIText command: {exc}")
            await ctx.respond(
                f"An unexpected error occurred: {str(exc)}",
                flags=hikari.MessageFlag.EPHEMERAL,
            )


@ai_group.register()
class AITextWithImage(
    lightbulb.SlashCommand, name="with_image", description="Generate text with an image"
):
    request: str = lightbulb.string("request", "The request to send to the AI.")
    image: hikari.Attachment = lightbulb.attachment("image", "The image to send to the AI.")
    model: Optional[str] = lightbulb.string(
        "model",
        "The model to use.",
        default=DEFAULT_MODEL,
        autocomplete=model_autocomplete,
    )
    prompt: Optional[str] = lightbulb.string(
        "prompt",
        "The prompt or preset to use.",
        default="default",
        autocomplete=prompt_preset_autocomplete,
    )

    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> Optional[hikari.Message | hikari.Snowflake]:
        await ctx.defer(ephemeral=False)

        # Verify image format
        try:
            Image.open(io.BytesIO(await self.image.read()))
        except (UnidentifiedImageError, IOError):
            embed = hikari.Embed(
                title="Error",
                description="Please upload a valid image file.",
                color=hikari.Color.from_hex_code("#ed4245")
            )
            return await ctx.respond(
                embed=embed,
                flags=hikari.MessageFlag.EPHEMERAL,
            )

        # Resolve prompt preset if needed
        resolved_prompt = PROMPT_PRESETS.get(self.prompt, self.prompt)

        try:
            image_data = io.BytesIO(await self.image.read())

            response, status = await generate_text(
                self.request, self.model, resolved_prompt, ctx.interaction.user.id, image_data
            )

            if status == 200:
                # Handle long responses
                if len(response) > MAX_DISCORD_MESSAGE_LENGTH:
                    chunks = []
                    while response:
                        split_idx = response.rfind("\n\n", 0, MAX_DISCORD_MESSAGE_LENGTH)

                        if split_idx in [-1, 0]:
                            split_idx = MAX_DISCORD_MESSAGE_LENGTH

                        chunk = response[:split_idx].rstrip()
                        response = response[split_idx:].lstrip()
                        chunks.append(chunk)
                    view = AIView(chunks)
                    await ctx.respond(chunks[0], components=view)
                    inter_client.start_view(view)
                else:
                    await ctx.respond(response)
            else:
                embed = hikari.Embed(
                    title="Error",
                    description=f"An error occurred:\n```py\n{str(response)}\n```",
                    color=hikari.Color.from_hex_code("#ed4245"),
                )
                embed.set_footer(f"Status code {status}")

                await ctx.respond(
                    embed=embed,
                    flags=hikari.MessageFlag.EPHEMERAL,
                )
        except Exception as exc:
            logger.error(f"Error in AITextWithImage command: {exc}")
            embed = hikari.Embed(
                title="Error",
                description="An error occurred:\n```py\n{str(exc)}\n```",
                color=hikari.Color.from_hex_code("#ed4245")
            )
            await ctx.respond(
                embed=embed,
                flags=hikari.MessageFlag.EPHEMERAL,
            )


@ai_group.register()
class AIImage(
    lightbulb.SlashCommand, name="image", description="Generate image from prompt"
):
    prompt: str = lightbulb.string("prompt", "The prompt to send to the AI.")

    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        await ctx.defer(ephemeral=False)

        try:
            image, status = await AIService.generate_image(self.prompt)

            if status == 200:
                await ctx.respond(attachments=[image])
            else:
                embed = hikari.Embed(
                    title="Error",
                    description=f"An error occurred:\n```py\n{str(image)}\n```",
                    color=hikari.Color.from_hex_code("#ed4245"),
                )
                embed.set_footer(f"Status code {status}")

                await ctx.respond(
                    embed=embed,
                    flags=hikari.MessageFlag.EPHEMERAL,
                )
        except Exception as exc:
            embed = hikari.Embed(
                title="Error",
                description=f"An unexpected error occurred:\n```py\n{str(exc)}\n```",
                color=hikari.Color.from_hex_code("#ed4245"),
            )
            await ctx.respond(
                embed=embed,
                flags=hikari.MessageFlag.EPHEMERAL,
            )


@ai_group.register()
class AIClear(
    lightbulb.SlashCommand, name="clear", description="Clear your chat history with the bot"
):
    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        user_id = ctx.interaction.user.id
        if user_id in chat_histories:
            del chat_histories[user_id]
            await ctx.respond(
                "Your chat history has been cleared.",
                flags=hikari.MessageFlag.EPHEMERAL,
            )
        else:
            await ctx.respond(
                "You don't have any chat history to clear.",
                flags=hikari.MessageFlag.EPHEMERAL,
            )


# Register commands and run bot
def main():
    client.register(ai_group)

    # Check required environment variables
    required_env_vars = [
        "DISCORD_BOT_TOKEN",
        "GEMINI_API_TOKEN",
        "GROQ_API_TOKEN",
        "IMAGE_GEN_TOKEN"
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        return

    # Start the bot
    logger.info("Starting bot")
    bot.run()


if __name__ == "__main__":
    main()
