import datetime
from dataclasses import dataclass

import aiosqlite

import hikari
import lightbulb

from .config import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MODEL,
    IMAGE_MODEL_TOKEN_COSTS,
    TEXT_MODEL_TOKEN_COSTS,
    WEEKLY_TOKEN_ALLOWANCE_PER_USER,
    logs_db_file,
    logger,
)
from .service import get_owner_ids


class UsageLimitExceeded(Exception):
    def __init__(self, required_tokens: int, remaining_tokens: int, reset_at: int):
        super().__init__("weekly token allowance exceeded")
        self.required_tokens = required_tokens
        self.remaining_tokens = remaining_tokens
        self.reset_at = reset_at


@dataclass(frozen=True)
class TokenUsageStatus:
    used_tokens: int
    remaining_tokens: int
    reset_at: int


class OwnerOnlyCommand(Exception):
    pass


def get_current_week_start(now: datetime.datetime | None = None) -> datetime.date:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return (now - datetime.timedelta(days=now.weekday())).date()


def get_next_week_reset_timestamp(week_start: datetime.date) -> int:
    reset_at = datetime.datetime.combine(
        week_start + datetime.timedelta(days=7),
        datetime.time.min,
        tzinfo=datetime.timezone.utc,
    )
    return int(reset_at.timestamp())


def get_text_token_cost(model: str | None) -> int:
    selected_model = model or DEFAULT_MODEL
    return TEXT_MODEL_TOKEN_COSTS.get(
        selected_model, TEXT_MODEL_TOKEN_COSTS[DEFAULT_MODEL]
    )


def get_image_token_cost(model: str | None) -> int:
    selected_model = model or DEFAULT_IMAGE_MODEL
    return IMAGE_MODEL_TOKEN_COSTS.get(
        selected_model, IMAGE_MODEL_TOKEN_COSTS[DEFAULT_IMAGE_MODEL]
    )


def get_command_token_cost(ctx: lightbulb.Context) -> int:
    interaction = ctx.interaction
    subcommand: str | None = None
    selected_model: str | None = None

    if interaction.options:
        option = interaction.options[0]
        if option.type == hikari.OptionType.SUB_COMMAND:
            subcommand = option.name
            for sub_option in option.options or []:
                if sub_option.name == "model" and isinstance(sub_option.value, str):
                    selected_model = sub_option.value
                    break

    if subcommand == "image":
        return get_image_token_cost(selected_model or DEFAULT_IMAGE_MODEL)
    if subcommand in {"text", "with_image"}:
        return get_text_token_cost(selected_model or DEFAULT_MODEL)
    return 0


async def get_token_usage_status(uid: str) -> TokenUsageStatus:
    week_start = get_current_week_start()
    reset_at = get_next_week_reset_timestamp(week_start)

    async with aiosqlite.connect(logs_db_file) as db:
        async with db.execute(
            """
            SELECT used_tokens
            FROM weekly_token_usage
            WHERE week_start = ? AND uid = ?
            """,
            (week_start.isoformat(), uid),
        ) as cur:
            row = await cur.fetchone()

    used_tokens = int(row[0]) if row else 0
    remaining_tokens = max(0, WEEKLY_TOKEN_ALLOWANCE_PER_USER - used_tokens)
    return TokenUsageStatus(
        used_tokens=used_tokens,
        remaining_tokens=remaining_tokens,
        reset_at=reset_at,
    )


async def ensure_token_allowance(uid: int | str, required_tokens: int) -> None:
    if int(uid) in await get_owner_ids():
        logger.info("Bot owner detected - ignoring token limits")
        return

    usage = await get_token_usage_status(str(uid))
    logger.info(
        "User %s - %s/%s weekly tokens used",
        uid,
        usage.used_tokens,
        WEEKLY_TOKEN_ALLOWANCE_PER_USER,
    )

    if usage.remaining_tokens < required_tokens:
        raise UsageLimitExceeded(
            required_tokens=required_tokens,
            remaining_tokens=usage.remaining_tokens,
            reset_at=usage.reset_at,
        )


async def ensure_owner(uid: int | str) -> None:
    if int(uid) not in await get_owner_ids():
        raise OwnerOnlyCommand()


async def is_owner(uid: int | str) -> bool:
    return int(uid) in await get_owner_ids()


@lightbulb.hook(
    lightbulb.ExecutionSteps.CHECKS, skip_when_failed=True, name="usage_limit"
)
async def usage_limit(_: lightbulb.ExecutionPipeline, ctx: lightbulb.Context):
    required_tokens = get_command_token_cost(ctx)
    if required_tokens <= 0:
        return

    await ensure_token_allowance(ctx.user.id, required_tokens)


@lightbulb.hook(
    lightbulb.ExecutionSteps.CHECKS, skip_when_failed=True, name="owner_only"
)
async def owner_only(_: lightbulb.ExecutionPipeline, ctx: lightbulb.Context):
    await ensure_owner(ctx.user.id)


async def increase_usage_limit(uid: int, used_tokens: int):
    week_start = get_current_week_start()

    async with aiosqlite.connect(logs_db_file) as db:
        async with db.execute(
            """
            INSERT INTO weekly_token_usage (week_start, uid, used_tokens, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(week_start, uid)
            DO UPDATE SET
                used_tokens = used_tokens + excluded.used_tokens,
                updated_at = CURRENT_TIMESTAMP
            RETURNING used_tokens
            """,
            (
                week_start.isoformat(),
                str(uid),
                used_tokens,
            ),
        ) as cur:
            row = await cur.fetchone()
        await db.commit()
        return int(row[0])


async def set_used_tokens(uid: int | str, used_tokens: int) -> int:
    week_start = get_current_week_start()
    normalized_used_tokens = max(0, min(WEEKLY_TOKEN_ALLOWANCE_PER_USER, used_tokens))

    async with aiosqlite.connect(logs_db_file) as db:
        await db.execute(
            """
            INSERT INTO weekly_token_usage (week_start, uid, used_tokens, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(week_start, uid)
            DO UPDATE SET
                used_tokens = excluded.used_tokens,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                week_start.isoformat(),
                str(uid),
                normalized_used_tokens,
            ),
        )
        await db.commit()
    return normalized_used_tokens


async def adjust_user_tokens(
    uid: int | str, action: str, amount: int = 0
) -> TokenUsageStatus:
    current_usage = await get_token_usage_status(str(uid))

    if action == "reset":
        new_used_tokens = 0
    elif action == "add":
        new_used_tokens = current_usage.used_tokens - amount
    elif action == "remove":
        new_used_tokens = current_usage.used_tokens + amount
    else:
        raise ValueError(f"unsupported token action: {action}")

    await set_used_tokens(uid, new_used_tokens)
    return await get_token_usage_status(str(uid))
