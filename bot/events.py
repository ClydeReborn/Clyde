import uuid
import logging
import traceback
import re
import datetime

import lightbulb
import hikari
import miru

import aiosqlite

from internal.config import (
    DEFAULT_PROMPT,
    DEFAULT_MODEL,
    MAX_DISCORD_MESSAGE_LENGTH,
    WEEKLY_TOKEN_ALLOWANCE_PER_USER,
    stats_db_file,
    data_logger,
    logger,
)
from internal.handlers import init_stats_db, init_logs_db
from internal.limits import (
    OwnerOnlyCommand,
    UsageLimitExceeded,
    ensure_token_allowance,
    get_text_token_cost,
    increase_usage_limit,
)
from internal.maintenance import MaintenanceModeActive, ensure_maintenance_access
from internal.service import AIService

loader = lightbulb.Loader()


@loader.error_handler
async def on_command_error(
    exc: lightbulb.exceptions.ExecutionPipelineFailedException,
) -> bool:
    ctx = exc.context

    if isinstance(exc.__cause__, lightbulb.prefab.cooldowns.OnCooldown):
        retry_after = int(exc.__cause__.remaining)
        em = hikari.Embed(
            title=":clock3: On Cooldown",
            description="Please wait a moment.\n"
            f"You can use this command again in **{retry_after}**s",
            color=hikari.Color.from_hex_code("#5865f2"),
        )
        await ctx.respond(em, flags=64)
        return True
    elif isinstance(exc.__cause__, UsageLimitExceeded):
        e = hikari.Embed(
            title="Weekly Token Allowance Reached",
            description=(
                f"This request costs **{exc.__cause__.required_tokens}** token(s), but "
                f"you only have **{exc.__cause__.remaining_tokens}** left out of "
                f"**{WEEKLY_TOKEN_ALLOWANCE_PER_USER}** this week.\n"
                f"Your balance resets <t:{exc.__cause__.reset_at}:R>."
            ),
            color=hikari.Color.from_hex_code("#5865f2"),
        )
        await ctx.respond(embed=e, flags=64)
        return True
    elif isinstance(exc.__cause__, OwnerOnlyCommand):
        e = hikari.Embed(
            title="Owner Only",
            description="Only bot owners can use this command.",
            color=hikari.Color.from_hex_code("#5865f2"),
        )
        await ctx.respond(embed=e, flags=64)
        return True
    elif isinstance(exc.__cause__, MaintenanceModeActive):
        end_text = (
            f"\nEnds: <t:{exc.__cause__.end_at}:R>."
            if exc.__cause__.end_at is not None
            else "\nEnds: This maintenance window is currently permanent."
        )
        e = hikari.Embed(
            title="Maintenance Mode",
            description=f"{exc.__cause__.message}{end_text}",
            color=hikari.Color.from_hex_code("#5865f2"),
        )
        await ctx.respond(embed=e, flags=64)
        return True

    errid = uuid.uuid4()
    em = hikari.Embed(
        title="<:error:1368156499167150171> Error",
        description=f"An internal error has occurred, and the request could not be fulfilled.\n"
        "The error has been raised in the logs.\n"
        f"To get details, report this error in the support server with the code `{errid}`.",
        color=hikari.Color.from_hex_code("#ed4245"),
    )
    logging.error(f"Error ID: {errid}")
    logging.error("".join(traceback.format_exception(exc.__cause__)))
    button_view = miru.View()
    button_view.add_item(
        miru.LinkButton(label="Support Server", url="https://discord.gg/E9UwEAPgU6")
    )

    await ctx.respond(em, flags=64, components=button_view.build())
    return True


@loader.listener(hikari.MessageCreateEvent)
async def on_message(event, bot: hikari.GatewayBot):
    content = event.message.content
    if isinstance(event, hikari.GuildMessageCreateEvent):
        channel = event.get_channel()
    else:
        channel = event.author

    reply = event.message.referenced_message

    if (
        isinstance(content, str)
        and (
            re.findall(rf"<@(!)?{bot.cache.get_me().id}>", content)
            or (reply and reply.author.id == bot.cache.get_me().id)
        )
        and not event.author.is_bot
    ):
        async with bot.rest.trigger_typing(channel):
            try:
                await ensure_maintenance_access(event.author.id)
            except MaintenanceModeActive as exc:
                end_text = (
                    f"\nEnds: <t:{exc.end_at}:R>."
                    if exc.end_at is not None
                    else "\nEnds: This maintenance window is currently permanent."
                )
                return await channel.send(f"{exc.message}{end_text}")

            cleaned_content = re.sub(rf"<@(!)?{bot.cache.get_me().id}>", "", content)
            token_cost = get_text_token_cost(DEFAULT_MODEL)
            try:
                await ensure_token_allowance(event.author.id, token_cost)
            except UsageLimitExceeded as exc:
                return await channel.send(
                    "You've used up your weekly token allowance. "
                    f"This reply needs {exc.required_tokens} token(s), "
                    f"and your balance resets <t:{exc.reset_at}:R>."
                )

            if isinstance(event, hikari.GuildMessageCreateEvent):
                filled_prompt = DEFAULT_PROMPT.format(
                    "Lunal",
                    event.get_channel().name,
                    event.get_guild().name,
                    datetime.datetime.now(),
                )
            else:
                filled_prompt = DEFAULT_PROMPT.format(
                    "Lunal", "", "", datetime.datetime.now()
                )

            message = await AIService.generate_text_with_gemini(
                cleaned_content,
                DEFAULT_MODEL,
                filled_prompt,
                event.message.author.id,
                None,
            )

            message += "\n\n-# Mention to continue this conversation."

            if len(message) > MAX_DISCORD_MESSAGE_LENGTH:
                return await channel.send(
                    "The response was too long.\nUse the slash command to see longer responses."
                )

            await increase_usage_limit(event.author.id, token_cost)
            return await channel.send(message)
    else:
        return None


@loader.task(lightbulb.uniformtrigger(hours=24))
async def record_stats(bot: hikari.GatewayBot):
    application = await bot.rest.fetch_application()
    guild_count = application.approximate_guild_count
    member_count = sum(
        g.member_count for g in bot.cache.get_guilds_view().values() if g.member_count
    )

    today = datetime.date.today().isoformat()
    async with aiosqlite.connect(stats_db_file) as db:
        await db.execute(
            "REPLACE INTO stats (date, guild_count, member_count) VALUES (?, ?, ?)",
            (today, guild_count, member_count),
        )
        await db.commit()
    data_logger.info(f"[STATS] {today}: {guild_count} guilds, {member_count} members")


@loader.listener(hikari.StartingEvent)
async def on_starting(_: hikari.StartingEvent, client: lightbulb.Client) -> None:
    """Handle bot startup event"""
    await client.start()
    logger.info("Shard initialization in progress")


@loader.listener(hikari.StartedEvent)
async def on_started(_: hikari.StartedEvent, bot: hikari.GatewayBot) -> None:
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
    await init_stats_db()
    await init_logs_db()
    my_name = bot.cache.get_me().username
    logger.info(
        f"{my_name} is now serving {all_servers} servers and {all_users} users on {all_shards} shard{'s' if all_shards > 1 else ''}"
    )


@loader.listener(hikari.ShardReadyEvent)
async def on_shard_started(ev: hikari.ShardReadyEvent) -> None:
    """Handle shard started event"""
    logger.info(f"Shard {ev.shard.id} is now ready")
