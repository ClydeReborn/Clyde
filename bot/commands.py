import datetime
import io
import platform
from typing import Optional

import hikari
import lightbulb
import miru

from PIL import Image, UnidentifiedImageError

from internal.limits import (
    adjust_user_tokens,
    get_image_token_cost,
    get_text_token_cost,
    get_token_usage_status,
    increase_usage_limit,
    is_owner,
    owner_only,
    usage_limit,
)
from internal.maintenance import (
    clear_maintenance_mode,
    get_maintenance_status,
    maintenance_check,
    parse_maintenance_end_date,
    set_maintenance_mode,
)
from .autocomplete import (
    model_autocomplete,
    prompt_preset_autocomplete,
    image_model_autocomplete,
    AIView,
)
from .modals import prompt_maintenance_modal, prompt_with_modal
from internal.config import (
    DEFAULT_MODEL,
    MAX_DISCORD_MESSAGE_LENGTH,
    PROMPT_PRESETS,
    DEFAULT_IMAGE_MODEL,
    MODELS,
    IMAGE_MODELS,
    WEEKLY_TOKEN_ALLOWANCE_PER_USER,
    chat_histories,
)
from internal.service import AIService, generate_text

loader = lightbulb.Loader()

ai_group = lightbulb.Group("ai", "AI command group")


@lightbulb.hook(lightbulb.ExecutionSteps.CHECKS)
async def is_banned(_: lightbulb.ExecutionPipeline, ctx: lightbulb.Context) -> None: ...


@ai_group.register()
class AIText(
    lightbulb.SlashCommand,
    name="text",
    description="Generate text with AI",
    hooks=[is_banned, maintenance_check, usage_limit],
):
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
    async def callback(self, ctx: lightbulb.Context, inter_client: miru.Client) -> None:
        modal_result = await prompt_with_modal(ctx, inter_client)
        if modal_result is None:
            return
        request = modal_result.request

        resolved_prompt = PROMPT_PRESETS.get(self.prompt, self.prompt).format(
            "Lunal", "", "", datetime.datetime.now()
        )

        response = await generate_text(
            request, self.model, resolved_prompt, ctx.interaction.user.id
        )

        if len(response) > MAX_DISCORD_MESSAGE_LENGTH:
            chunks = []
            while response:
                split_idx = response.rfind("\n\n", 0, MAX_DISCORD_MESSAGE_LENGTH)

                if split_idx in [-1, 0]:
                    split_idx = MAX_DISCORD_MESSAGE_LENGTH

                chunk = response[:split_idx].rstrip()
                response = response[split_idx:].lstrip()
                chunks.append(chunk)
            view = AIView(chunks, modal_result.context.interaction)
            await modal_result.context.edit_response(chunks[0], components=view.build())
            inter_client.start_view(view)
        else:
            await modal_result.context.edit_response(response)

        await increase_usage_limit(ctx.user.id, get_text_token_cost(self.model))


@ai_group.register()
class AITextWithImage(
    lightbulb.SlashCommand,
    name="with_image",
    description="Generate text with an image",
    hooks=[is_banned, maintenance_check, usage_limit],
):
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
        self, ctx: lightbulb.Context, inter_client: miru.Client
    ) -> Optional[hikari.Message | hikari.Snowflake]:
        modal_result = await prompt_with_modal(ctx, inter_client)
        if modal_result is None:
            return None
        request = modal_result.request

        try:
            Image.open(io.BytesIO(await self.image.read()))
        except (UnidentifiedImageError, IOError):
            embed = hikari.Embed(
                title="<:error:1368156499167150171> Error",
                description="Please upload a valid image file.",
                color=hikari.Color.from_hex_code("#ed4245"),
            )
            await modal_result.context.edit_response(embed=embed, components=[])
            return None

        resolved_prompt = PROMPT_PRESETS.get(self.prompt, self.prompt).format(
            "Lunal", "", "", datetime.datetime.now()
        )

        image_data = io.BytesIO(await self.image.read())

        response = await generate_text(
            request,
            self.model,
            resolved_prompt,
            ctx.interaction.user.id,
            image_data,
        )

        if len(response) > MAX_DISCORD_MESSAGE_LENGTH:
            chunks = []
            while response:
                split_idx = response.rfind("\n\n", 0, MAX_DISCORD_MESSAGE_LENGTH)

                if split_idx in [-1, 0]:
                    split_idx = MAX_DISCORD_MESSAGE_LENGTH

                chunk = response[:split_idx].rstrip()
                response = response[split_idx:].lstrip()
                chunks.append(chunk)
            view = AIView(chunks, interaction=modal_result.context.interaction)
            await modal_result.context.edit_response(chunks[0], components=view.build())
            inter_client.start_view(view)
        else:
            await modal_result.context.edit_response(response)

        return await increase_usage_limit(ctx.user.id, get_text_token_cost(self.model))


@ai_group.register()
class AIImage(
    lightbulb.SlashCommand,
    name="image",
    description="Generate image from prompt",
    hooks=[
        is_banned,
        maintenance_check,
        lightbulb.prefab.cooldowns.fixed_window(60, 1, "user"),
        usage_limit,
    ],
):
    prompt: str = lightbulb.string("prompt", "The image to generate.")
    model: Optional[str] = lightbulb.string(
        "model",
        "The model to use.",
        default=DEFAULT_IMAGE_MODEL,
        autocomplete=image_model_autocomplete,
    )

    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        await ctx.defer(ephemeral=False)

        image = await AIService.generate_image(
            self.model, self.prompt, str(ctx.user.id)
        )

        if isinstance(image, io.BytesIO):  # image generated
            # Discord may compress WebP images badly, convert them to PNG.
            orig = Image.open(image)
            png = io.BytesIO()
            orig.save(png, format="PNG")
            png.seek(0)

            await ctx.respond(attachments=[hikari.Bytes(png, "image.png")])
        else:  # blocked by security
            await ctx.respond(image)

        await increase_usage_limit(ctx.user.id, get_image_token_cost(self.model))


@ai_group.register()
class AIClear(
    lightbulb.SlashCommand,
    name="clear",
    description="Clear your chat history with the bot",
    hooks=[is_banned, maintenance_check],
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


@loader.command
class Info(
    lightbulb.SlashCommand,
    name="info",
    description="Display information about the bot",
    hooks=[maintenance_check],
):
    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context, bot: hikari.GatewayBot) -> None:
        models_length = sum(len(models) for models in MODELS.values()) + len(
            IMAGE_MODELS.keys()
        )
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


@loader.command
class Ping(
    lightbulb.SlashCommand,
    name="ping",
    description="Ping the bot",
    hooks=[maintenance_check],
):
    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context, bot: hikari.GatewayBot) -> None:
        latency = round(bot.heartbeat_latency * 1000)
        pe = hikari.Embed(
            description=f"**{latency}**ms", color=hikari.Color.from_hex_code("#5865F2")
        )
        await ctx.respond(embed=pe)


@loader.command
class Invite(
    lightbulb.SlashCommand,
    name="invite",
    description="Invite the bot to your server",
    hooks=[maintenance_check],
):
    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context, bot: hikari.GatewayBot) -> None:
        my_id = bot.cache.get_me().id
        button_view = miru.View()
        button_view.add_item(
            miru.LinkButton(
                label="Invite",
                url=f"https://discord.com/api/oauth2/authorize?client_id={my_id}&scope=bot+applications.commands",
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


@loader.command
class Tokens(
    lightbulb.SlashCommand,
    name="tokens",
    description="Manage a user's weekly token balance",
):
    action: str = lightbulb.string(
        "action",
        "Whether to view, add, remove, or reset tokens.",
        choices=[
            lightbulb.commands.options.Choice(name="view", value="view"),
            lightbulb.commands.options.Choice(name="add", value="add"),
            lightbulb.commands.options.Choice(name="remove", value="remove"),
            lightbulb.commands.options.Choice(name="reset", value="reset"),
        ],
    )
    user: Optional[hikari.User] = lightbulb.user(
        "user",
        "The user whose tokens to manage or view.",
        default=None,
    )
    amount: Optional[int] = lightbulb.integer(
        "amount",
        "Number of tokens to add or remove.",
        default=None,
        min_value=1,
    )

    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context) -> None:
        target_user = self.user or ctx.user
        requester_is_owner = await is_owner(ctx.user.id)

        if self.action == "view":
            if target_user.id != ctx.user.id and not requester_is_owner:
                return await ctx.respond(
                    "You can only view your own token balance.",
                    flags=hikari.MessageFlag.EPHEMERAL,
                )

            if await is_owner(target_user.id):
                return await ctx.respond(
                    f"{target_user.mention} has unlimited tokens as a bot owner.",
                    flags=hikari.MessageFlag.EPHEMERAL,
                )

            usage = await get_token_usage_status(str(target_user.id))
            return await ctx.respond(
                f"{target_user.mention} has **{usage.remaining_tokens}/"
                f"{WEEKLY_TOKEN_ALLOWANCE_PER_USER}** tokens remaining this week.\n"
                f"Used: **{usage.used_tokens}/{WEEKLY_TOKEN_ALLOWANCE_PER_USER}**.\n"
                f"Next reset: <t:{usage.reset_at}:R>.",
                flags=hikari.MessageFlag.EPHEMERAL,
            )

        if not requester_is_owner:
            return await ctx.respond(
                embed=hikari.Embed(
                    title="Owner Only",
                    description="Only bot owners can manage token balances.",
                    color=hikari.Color.from_hex_code("#5865f2"),
                ),
                flags=hikari.MessageFlag.EPHEMERAL,
            )

        if self.action in {"add", "remove"} and self.amount is None:
            return await ctx.respond(
                "You need to provide an amount for `add` or `remove`.",
                flags=hikari.MessageFlag.EPHEMERAL,
            )

        updated_usage = await adjust_user_tokens(
            target_user.id,
            self.action,
            self.amount or 0,
        )

        if self.action == "reset":
            message = (
                f"Reset {target_user.mention}'s weekly tokens to "
                f"**{WEEKLY_TOKEN_ALLOWANCE_PER_USER}/{WEEKLY_TOKEN_ALLOWANCE_PER_USER}**.\n"
                f"Next reset: <t:{updated_usage.reset_at}:R>."
            )
        else:
            action_word = "Added" if self.action == "add" else "Removed"
            direction = "to" if self.action == "add" else "from"
            message = (
                f"{action_word} **{self.amount}** token(s) {direction} {target_user.mention}.\n"
                f"They now have **{updated_usage.remaining_tokens}/"
                f"{WEEKLY_TOKEN_ALLOWANCE_PER_USER}** tokens remaining this week.\n"
                f"Next reset: <t:{updated_usage.reset_at}:R>."
            )

        await ctx.respond(message, flags=hikari.MessageFlag.EPHEMERAL)


@loader.command
class Maintenance(
    lightbulb.SlashCommand,
    name="maintenance",
    description="Manage bot maintenance mode",
    hooks=[owner_only],
):
    action: str = lightbulb.string(
        "action",
        "Whether to start, stop, or view maintenance mode.",
        choices=[
            lightbulb.commands.options.Choice(name="start", value="start"),
            lightbulb.commands.options.Choice(name="stop", value="stop"),
            lightbulb.commands.options.Choice(name="status", value="status"),
        ],
    )
    end_date: Optional[str] = lightbulb.string(
        "end_date",
        "Optional UTC end date, e.g. 2026-07-10 or 2026-07-10 18:00.",
        default=None,
    )

    @lightbulb.invoke
    async def callback(self, ctx: lightbulb.Context, inter_client: miru.Client) -> None:
        if self.action == "stop":
            await clear_maintenance_mode()
            return await ctx.respond(
                "Maintenance mode has been disabled.",
                flags=hikari.MessageFlag.EPHEMERAL,
            )

        if self.action == "status":
            status = await get_maintenance_status()
            if not status.active:
                return await ctx.respond(
                    "Maintenance mode is currently disabled.",
                    flags=hikari.MessageFlag.EPHEMERAL,
                )

            end_text = (
                f"<t:{status.end_at}:F> (<t:{status.end_at}:R>)"
                if status.end_at is not None
                else "Permanent"
            )
            started_text = (
                f"<t:{status.started_at}:F>"
                if status.started_at is not None
                else "Unknown"
            )
            return await ctx.respond(
                f"Maintenance mode is active.\n"
                f"Started: {started_text}\n"
                f"Ends: {end_text}\n"
                f"Message: {status.message}",
                flags=hikari.MessageFlag.EPHEMERAL,
            )

        parsed_end_date = None
        if self.end_date:
            try:
                parsed_end_date = parse_maintenance_end_date(self.end_date)
            except ValueError as exc:
                return await ctx.respond(
                    str(exc),
                    flags=hikari.MessageFlag.EPHEMERAL,
                )

            if parsed_end_date <= datetime.datetime.now(datetime.timezone.utc):
                return await ctx.respond(
                    "The maintenance end date must be in the future.",
                    flags=hikari.MessageFlag.EPHEMERAL,
                )

        modal_result = await prompt_maintenance_modal(ctx, inter_client)
        if modal_result is None:
            return

        await set_maintenance_mode(modal_result.reason, parsed_end_date)

        end_text = (
            f"<t:{int(parsed_end_date.timestamp())}:F> (<t:{int(parsed_end_date.timestamp())}:R>)"
            if parsed_end_date is not None
            else "Permanent"
        )
        await modal_result.context.edit_response(
            f"Maintenance mode enabled.\nEnds: {end_text}\nMessage: {modal_result.reason}",
        )


loader.command(ai_group)
