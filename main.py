import os
import io
import re
import datetime
import asyncio
import difflib
import logging
import contextlib
import platform
import time
import traceback
import uuid
from functools import wraps
from typing import Optional, Dict, List

import hikari
import lightbulb
import miru
import httpx
import aiosqlite
import aiocron
from google import genai
from google.genai import types
from groq import AsyncGroq
from groq.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lunalai")
data_logger = logging.getLogger("lunalai.stats")

# Load environment variables
load_dotenv()

# Constants
MAX_DISCORD_MESSAGE_LENGTH = 2000
MAX_CHAT_HISTORY = 30
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_PROMPT = """
You are {}, a Discord chatbot.

You are aware of the following:
- You are currently in the #{} channel.
- You are currently in the {} server.

If the placeholders haven't been replaced, disregard them.

Your personality reflects your name. Try to make up a personality that matches your name.
"""

# Model lists for autocomplete
MODELS = {
    "gemini": [
        # Gemini 2.5 series
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        # Gemini 2.0 series
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    "gemma": [
        # Gemma 2 models have been decommissioned.
        # Gemma 3 series
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3n-e2b-it",
        "gemma-3n-e4b-it"
    ],
    "other": [
        # Qwen models
        "qwen/qwen3-32b",
        # Llama 3.x series
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        # Llama 4 series
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-7b-16e-instruct",
        # Moonshot AI models
        "moonshotai/kimi-k2-instruct-0905",
        # OpenAI open source models
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # Groq in-house models
        "groq/compound",
        "groq/compound-mini"
    ],
}

# Prompt presets
PROMPT_PRESETS = {
    "default": DEFAULT_PROMPT,
    "gpt": "You are GPT-5.1, a large language model from OpenAI. Refer to yourself as ChatGPT.",
    "casual": "Respond casually, no need to use overly verbose responses.",
    "discord": "You are a Discord user, respond lowercase only, no punctuation, and treat yourself like you're on IRC.",
    "programmer": "You are a programming assistant who will provide suggestions to codes given and attempt to rework when asked.",
    "storyteller": "You will make creative stories about the given topics, tell the user said story.",
}

# Initialize bot
bot = hikari.GatewayBot(
    token=os.getenv("DISCORD_BOT_TOKEN"),
    intents=hikari.Intents.ALL_UNPRIVILEGED,
)
db_file = os.getenv("STATS_DB")
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


async def init_db():
    if not os.path.exists(db_file):
        async with aiosqlite.connect(db_file) as db:
            await db.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                date TEXT PRIMARY KEY,
                guild_count INTEGER,
                member_count INTEGER
            )
            """)

            await db.execute("""
            CREATE TABLE IF NOT EXISTS model_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model TEXT,
                guild_id INTEGER,
                tps REAL,
                ttft REAL,
                status TEXT,
                error_type TEXT,
                tokens INTEGER
            )
            """)  # TODO: record data here over time

            await db.commit()
        data_logger.info("[STATS] data DB created")

@client.error_handler
async def on_command_error(exc: lightbulb.exceptions.ExecutionPipelineFailedException) -> bool:
    ctx = exc.context
    errid = uuid.uuid4()
    em = hikari.Embed(
        title="<:error:1368156499167150171> Error",
        description=f"We were unable to generate your response.\n"
        f"Please report this in the support server with the following code: {errid}",
        color=hikari.Color.from_hex_code("#ed4245"),
    )
    logging.error(f"Error ID: {errid}")
    logging.error("".join(traceback.format_exception(exc.__cause__)))
    button_view = miru.View()
    button_view.add_item(
        miru.LinkButton(
            label="Support Server",
            url="https://discord.gg/E9UwEAPgU6"
        )
    )

    await ctx.respond(em, flags=64, components=button_view.build())
    return True

@bot.listen(hikari.MessageCreateEvent)
async def on_message(event):
    content = event.message.content
    if isinstance(event, hikari.GuildMessageCreateEvent):
        channel = event.get_channel()
    else:
        channel = event.author

    reply = event.message.referenced_message

    if isinstance(content, str) and (
        re.findall(fr'<@(!)?{event.app.cache.get_me().id}>', content) or (
            reply and reply.author.id == event.app.cache.get_me().id
        )
    ) and not event.author.is_bot:
        async with bot.rest.trigger_typing(channel):
            cleaned_content = re.sub(fr'<@(!)?{event.app.cache.get_me().id}>', '', content)

            if isinstance(event, hikari.GuildMessageCreateEvent):
                filled_prompt = DEFAULT_PROMPT.format("Lunal", event.get_channel().name, event.get_guild().name)
            else:
                filled_prompt = DEFAULT_PROMPT.format("Lunal", "", "")

            message = await AIService.generate_text_with_gemini(
                cleaned_content, DEFAULT_MODEL, filled_prompt, event.message.author.id, None
            )

            message += "\n\n-# Mention to continue this conversation."

            if len(message) > MAX_DISCORD_MESSAGE_LENGTH:
                return await channel.send("The response was too long.\nUse the slash command to see longer responses.")

            await channel.send(message)

async def record_stats():
    application = await bot.rest.fetch_application()
    guild_count = application.approximate_guild_count
    member_count = sum(
        g.member_count for g in bot.cache.get_guilds_view().values() if g.member_count
    )

    today = datetime.date.today().isoformat()
    async with aiosqlite.connect(db_file) as db:
        await db.execute(
            "REPLACE INTO stats (date, guild_count, member_count) VALUES (?, ?, ?)",
            (today, guild_count, member_count),
        )
        await db.commit()
    data_logger.info(f"[STATS] {today}: {guild_count} guilds, {member_count} members")


class AIView(miru.View):
    def __init__(
        self, entries: list[str], interaction: hikari.CommandInteraction
    ) -> None:
        super().__init__(
            timeout=60
        )  # Increased timeout to 60 seconds is more user-friendly
        self.entries = entries
        self._interaction = interaction
        self.index = 0

        # Initialize buttons with proper disabled states
        self.previous_page.disabled = True
        self.next_page.disabled = len(entries) <= 1

    @miru.button(emoji="<:left:1368155093337243748>", style=hikari.ButtonStyle.SECONDARY)
    async def previous_page(self, ctx: miru.ViewContext, button: miru.Button) -> None:
        self.index -= 1
        self._update_buttons()
        await ctx.edit_response(self.entries[self.index], components=self.build())

    @miru.button(emoji="<:right:1368155064925163550>", style=hikari.ButtonStyle.SECONDARY)
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
            await self._interaction.edit_initial_response(components=self.build())

def exponential(retry_cnt: int, retry_min: int, retry_max: int):
    """Exponentially back off on failure."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                got_exc = None
                for attempt in range(retry_cnt):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        got_exc = exc
                        await asyncio.sleep(min(retry_min * (2 ** attempt), retry_max))
                raise got_exc
            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            got_exc = None
            for attempt in range(retry_cnt):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    got_exc = exc
                    time.sleep(min(retry_min * (2 ** attempt), retry_max))
            raise got_exc
        return sync_wrapper
    return decorator

class AIService:
    """Service for handling AI-related operations"""

    @staticmethod
    def format_chat_history(messages: List[str]) -> list[dict]:
        """Format chat history for AI consumption"""
        formatted = []
        for i, message in enumerate(messages[:MAX_CHAT_HISTORY], start=1):
            if i % 2 == 0:
                formatted.append({"role": "user", "content": message})
            else:
                formatted.append({"role": "assistant", "content": message})
        return formatted

    @staticmethod
    @exponential(3, 5, 30)
    async def check_llamaguard(text: str) -> tuple[bool, int]:
        """Validate safety of a request or response"""

        logger.info("Dispatching text to Llama Guard")
        request = await groq_client.chat.completions.create(
            model="meta-llama/llama-guard-4-12b",
            messages=[
                ChatCompletionUserMessageParam(role="user", content=text)
            ],
            temperature=0,
        )

        raw = request.choices[0].message.content.strip().lower()
        parts = raw.split()

        tag = parts[0] if parts else "unsafe"
        code = parts[1].upper() if len(parts) > 1 else None

        severity_map = {
            "S1": 3,  # violent crimes
            "S2": 3,  # non-violent crimes
            "S3": 3,  # sex-related crimes
            "S4": 3,  # child sexual exploitation
            "S5": 2,  # defamation
            "S6": 1,  # specialized advice
            "S7": 2,  # privacy
            "S8": 2,  # intellectual property
            "S9": 3,  # indiscriminate weapons
            "S10": 3,  # hate speech
            "S11": 3,  # suicide & self-harm
            "S12": 2,  # sexual content
            "S13": 2,  # elections
            "S14": 2,  # code interpreter abuse
            None: 0,  # no violation
        }

        severity = severity_map.get(code, 0)
        is_safe = (tag == "safe")

        logger.info(f"Verdict: {tag}, {code}")

        if tag == "unsafe" and code is None:
            return False, 2
            
        return is_safe, severity

    @staticmethod
    @exponential(3, 5, 30)
    async def generate_text_with_gemini(
        request: str,
        model: str,
        system_prompt: str,
        user_id: int,
        image: Optional[io.BytesIO] = None,
    ) -> Optional[str]:
        """Generate text using Gemini models with native chat history"""
        # Check safety of input
        is_safe, severity = await AIService.check_llamaguard(request)
        if not is_safe and severity >= 2:
            return "I'm sorry, but I'm unable to assist with that."
        
        # Initialize user chat history if missing
        chat_histories.setdefault(user_id, [])

        # System instruction (not stored in history; just config)
        config = types.GenerateContentConfig(system_instruction=system_prompt)

        # Build structured chat history for Gemini
        history = []
        for i, msg in enumerate(chat_histories[user_id]):
            role = "user" if i % 2 == 0 else "model"
            history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg)])
            )

        # Add current user request
        if image:
            parts = [
                types.Part.from_text(text=request),
                types.Part.from_bytes(data=image.read(), mime_type="image/png"),
            ]
        else:
            parts = [types.Part.from_text(text=request)]

        history.append(types.Content(role="user", parts=parts))

        # Choose API call depending on model
        if "gemma" in model.lower():
            response = await gemini_client.aio.models.generate_content(
                model=model, contents=history
            )
        else:
            response = await gemini_client.aio.models.generate_content(
                model=model, config=config, contents=history
            )

        result = response.text

        is_safe, severity = await AIService.check_llamaguard(result)
        if not is_safe and severity >= 2:
            result = "Sorry, that's beyond my current scope."
        

        # Store assistant reply in history
        chat_histories[user_id].append(request)  # user input
        chat_histories[user_id].append(result)   # assistant output

        return result

    @staticmethod
    @exponential(3, 5, 30)
    async def generate_text_with_groq(
        request: str, model: str, system_prompt: str, user_id: int
    ) -> Optional[str]:
        """Generate text using Groq models"""
        is_safe, severity = await AIService.check_llamaguard(request)
        if not is_safe and severity >= 2:
            return "I'm sorry, but I'm unable to assist with that."
        
        chat_histories.setdefault(user_id, []).append(request)
        formatted_history = AIService.format_chat_history(chat_histories[user_id])

        response = await groq_client.chat.completions.create(
            model=model,
            messages=[
                ChatCompletionSystemMessageParam(role="system", content="system_prompt"),
                *formatted_history,
            ],
        )

        result = response.choices[0].message.content

        is_safe, severity = await AIService.check_llamaguard(result)
        if not is_safe and severity >= 2:
            result = "Sorry, that's beyond my current scope."

        # Models like Qwen 3 generate thinking, discard it from the response.
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)

        chat_histories[user_id].append(result)
        return result

    @staticmethod
    async def generate_image(prompt: str) -> Optional[io.BytesIO | str]:
        """Generate an image from text prompt"""
        is_safe, severity = await AIService.check_llamaguard(prompt)
        if not is_safe and severity >= 2:
            return "I'm sorry, but I cannot generate this image."
        
        async with httpx.AsyncClient(timeout=60.0) as img_client:
            response = await img_client.post(
                "https://ir-api.myqa.cc/v1/openai/images/generations",
                json={
                    "model": "openai/gpt-image-1.5:free",
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
                raise RuntimeError("Empty response from image service")

            image_data = await img_client.get(data["data"][0]["url"])
            image = io.BytesIO(image_data.content)
            image.seek(0)

            logging.info(f"MIME type: {Image.MIME.get(image.format)}")

            return image

async def generate_text(
    request: str,
    model: str,
    prompt: str,
    user_id: int,
    image: Optional[io.BytesIO] = None,
) -> Optional[str]:
    """Unified text generation function that routes to appropriate service"""
    try:
        # Route to appropriate service based on model name
        if any(name in model.lower() for name in ["gemma", "gemini"]):
            return await AIService.generate_text_with_gemini(
                request, model, prompt, user_id, image
            )
        if image:
            raise ValueError("Image processing not supported for non-Gemini models")
        return await AIService.generate_text_with_groq(request, model, prompt, user_id)
    except Exception as exc:
        logger.error(f"Text generation error: {exc}")
        raise RuntimeError(f"An unexpected error occurred: {str(exc)}")


# Event handlers
async def on_starting(_: hikari.StartingEvent) -> None:
    """Handle bot startup event"""
    await client.start()
    logger.info("Shard initialization in progress")


bot.subscribe(hikari.StartingEvent, on_starting)


async def on_started(_: hikari.StartedEvent) -> None:
    """Handle bot started event"""
    application = await bot.rest.fetch_application()                           
    all_servers = application.approximate_guild_count
    all_users = application.approximate_user_install_count
    all_shards = bot.shard_count
    await bot.update_presence(
        status=hikari.Status.IDLE,
        activity=hikari.Activity(
            type=hikari.ActivityType.LISTENING, name="user requests - / for commands"
        ),
    )
    await init_db()
    aiocron.crontab("0 0 * * *", func=record_stats, loop=asyncio.get_running_loop())
    logger.info(
        f"Lunal is now serving {all_servers} servers and {all_users} users on {all_shards} shard{'s' if all_shards > 1 else ''}"
    )


async def on_shard_started(ev: hikari.ShardReadyEvent) -> None:
    """Handle shard started event"""
    logger.info(f"Shard {ev.shard}/{ev.shard.shard_count} is now ready")


bot.subscribe(hikari.StartedEvent, on_started)


# Autocomplete handlers
async def model_autocomplete(ctx: lightbulb.AutocompleteContext) -> None:
    """Provide model autocomplete suggestions"""
    current_value: str = ctx.focused.value or ""

    # Flatten the model list
    all_models = [model for category in MODELS.values() for model in category]

    # Direct matches first
    prefix_matches = [
        m for m in all_models if m.lower().startswith(current_value.lower())
    ]
    if prefix_matches:
        return await ctx.respond(prefix_matches[:25])

    # Then fuzzy matches
    fuzzy_matches = difflib.get_close_matches(
        current_value, all_models, n=10, cutoff=0.3
    )
    if fuzzy_matches:
        return await ctx.respond(fuzzy_matches)

    # Default suggestions grouped by type
    suggestions = []
    for category, models in MODELS.items():
        suggestions.extend(models[:3])  # Take first 3 from each category

    return await ctx.respond(suggestions[:25])


async def prompt_preset_autocomplete(ctx: lightbulb.AutocompleteContext) -> None:
    """Provide prompt preset autocomplete suggestions"""
    current_value: str = ctx.focused.value or ""

    # Direct matches first
    prefix_matches = [
        k for k in PROMPT_PRESETS if k.lower().startswith(current_value.lower())
    ]
    if prefix_matches:
        return await ctx.respond(prefix_matches)

    # Then fuzzy matches
    fuzzy_matches = difflib.get_close_matches(
        current_value, PROMPT_PRESETS.keys(), n=5, cutoff=0.3
    )
    if fuzzy_matches:
        return await ctx.respond(fuzzy_matches)

    # Default is all presets
    return await ctx.respond(list(PROMPT_PRESETS.keys()))


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

        response = await generate_text(
            self.request, self.model, resolved_prompt, ctx.interaction.user.id
        )

        # Handle long responses
        if len(response) > MAX_DISCORD_MESSAGE_LENGTH:
            chunks = []
            while response:
                split_idx = response.rfind(
                    "\n\n", 0, MAX_DISCORD_MESSAGE_LENGTH
                )

                if split_idx in [-1, 0]:
                    split_idx = MAX_DISCORD_MESSAGE_LENGTH

                chunk = response[:split_idx].rstrip()
                response = response[split_idx:].lstrip()
                chunks.append(chunk)
            view = AIView(chunks, ctx.interaction)
            await ctx.respond(chunks[0], components=view)
            inter_client.start_view(view)
        else:
            await ctx.respond(response)


@ai_group.register()
class AITextWithImage(
    lightbulb.SlashCommand, name="with_image", description="Generate text with an image"
):
    request: str = lightbulb.string("request", "The request to send to the AI.")
    image: hikari.Attachment = lightbulb.attachment(
        "image", "The image to send to the AI."
    )
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
    async def callback(
        self, ctx: lightbulb.Context
    ) -> Optional[hikari.Message | hikari.Snowflake]:
        await ctx.defer(ephemeral=False)

        # Verify image format
        try:
            Image.open(io.BytesIO(await self.image.read()))
        except (UnidentifiedImageError, IOError):
            embed = hikari.Embed(
                title="<:error:1368156499167150171> Error",
                description="Please upload a valid image file.",
                color=hikari.Color.from_hex_code("#ed4245"),
            )
            return await ctx.respond(
                embed=embed,
                flags=hikari.MessageFlag.EPHEMERAL,
            )

        # Resolve prompt preset if needed
        resolved_prompt = PROMPT_PRESETS.get(self.prompt, self.prompt)

        image_data = io.BytesIO(await self.image.read())

        response  = await generate_text(
            self.request,
            self.model,
            resolved_prompt,
            ctx.interaction.user.id,
            image_data,
        )

        # Handle long responses
        if len(response) > MAX_DISCORD_MESSAGE_LENGTH:
            chunks = []
            while response:
                split_idx = response.rfind(
                    "\n\n", 0, MAX_DISCORD_MESSAGE_LENGTH
                )

                if split_idx in [-1, 0]:
                    split_idx = MAX_DISCORD_MESSAGE_LENGTH

                chunk = response[:split_idx].rstrip()
                response = response[split_idx:].lstrip()
                chunks.append(chunk)
            view = AIView(chunks, interaction=ctx.interaction)
            await ctx.respond(chunks[0], components=view)
            return inter_client.start_view(view)
        else:
            return await ctx.respond(response)


@ai_group.register()
class AIImage(
    lightbulb.SlashCommand, name="image", description="Generate image from prompt"
):
    prompt: str = lightbulb.string("prompt", "The prompt to send to the AI.")

    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        await ctx.defer(ephemeral=False)

        image = await AIService.generate_image(self.prompt)

        if isinstance(image, io.BytesIO):  # image generated
            await ctx.respond(attachments=[image])
        else:  # blocked by security
            await ctx.respond(image)


@ai_group.register()
class AIClear(
    lightbulb.SlashCommand,
    name="clear",
    description="Clear your chat history with the bot",
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


@client.register()
class Info(
    lightbulb.SlashCommand, name="info", description="Display information about the bot"
):
    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        models_length = sum(len(models) for models in MODELS.values())
        application = await bot.rest.fetch_application()
        guild_count = application.approximate_guild_count
        user_count = application.approximate_user_install_count

        ie = hikari.Embed(
            title=f"About {bot.get_me().display_name}",
            description=f"Serving {guild_count} servers and"
                        f" {user_count} users with AI for free",
            color=hikari.Color.from_hex_code("#5865F2"),
        )

        ie.add_field(name="Python Version", value=platform.python_version())
        ie.add_field(name="Hikari Version", value=hikari.__version__)
        ie.add_field(name="Models Available", value=str(models_length))

        await ctx.respond(embed=ie)


@client.register()
class Ping(
    lightbulb.SlashCommand,
    name="ping",
    description="Ping the bot",
):
    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        latency = round(bot.heartbeat_latency * 1000)
        pe = hikari.Embed(
            description=f"**{latency}**ms", color=hikari.Color.from_hex_code("#5865F2")
        )
        await ctx.respond(embed=pe)


@client.register()
class Invite(
    lightbulb.SlashCommand,
    name="invite",
    description="Invite the bot to your server",
):
    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        button_view = miru.View()
        button_view.add_item(
            miru.LinkButton(
                label="Invite",
                url="https://discord.com/api/oauth2/authorize?client_id=900004137742262332&scopes=bot+applications.commands",
            )
        )

        ie = hikari.Embed(
            description="Add me to your server by pressing the button below.",
            color=hikari.Color.from_hex_code("#5865f2"),
        )

        await ctx.respond(
            embed=ie,
            components=button_view.build(),
        )


# Register commands and run bot
def main():
    client.register(ai_group)

    # Check required environment variables
    required_env_vars = [
        "DISCORD_BOT_TOKEN",
        "GEMINI_API_TOKEN",
        "GROQ_API_TOKEN",
        "IMAGE_GEN_TOKEN",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        print(
            f"Error: Missing required environment variables: {', '.join(missing_vars)}"
        )
        return

    # Start the bot
    logger.info("Starting bot")
    bot.run()


if __name__ == "__main__":
    main()
