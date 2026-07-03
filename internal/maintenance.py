import datetime
from dataclasses import dataclass

import aiosqlite
import hikari
import lightbulb

from .config import logs_db_file
from .service import get_owner_ids


class MaintenanceModeActive(Exception):
    def __init__(self, message: str, end_at: int | None):
        super().__init__("maintenance mode is active")
        self.message = message
        self.end_at = end_at


@dataclass(frozen=True)
class MaintenanceStatus:
    active: bool
    message: str | None
    started_at: int | None
    end_at: int | None


def _parse_timestamp(value: str | None) -> int | None:
    if value is None:
        return None
    dt = datetime.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())


def parse_maintenance_end_date(value: str) -> datetime.datetime:
    normalized = value.strip()
    formats = (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    )

    for fmt in formats:
        try:
            parsed = datetime.datetime.strptime(normalized, fmt)
            if fmt == "%Y-%m-%d":
                parsed = parsed.replace(hour=23, minute=59, second=59)
            return parsed.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue

    raise ValueError(
        "Invalid end date format. Use `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` in UTC."
    )


async def get_maintenance_status() -> MaintenanceStatus:
    async with aiosqlite.connect(logs_db_file) as db:
        async with db.execute(
            """
            SELECT active, message, started_at, end_at
            FROM maintenance_mode
            WHERE id = 1
            """
        ) as cur:
            row = await cur.fetchone()

    if not row or int(row[0]) == 0:
        return MaintenanceStatus(False, None, None, None)

    end_at = _parse_timestamp(row[3])
    if end_at is not None and end_at <= int(
        datetime.datetime.now(datetime.timezone.utc).timestamp()
    ):
        await clear_maintenance_mode()
        return MaintenanceStatus(False, None, None, None)

    return MaintenanceStatus(
        active=True,
        message=row[1],
        started_at=_parse_timestamp(row[2]),
        end_at=end_at,
    )


async def set_maintenance_mode(message: str, end_at: datetime.datetime | None) -> None:
    started_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    end_at_value = end_at.replace(microsecond=0).isoformat() if end_at else None

    async with aiosqlite.connect(logs_db_file) as db:
        await db.execute(
            """
            INSERT INTO maintenance_mode (id, active, message, started_at, end_at, updated_at)
            VALUES (1, 1, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id)
            DO UPDATE SET
                active = 1,
                message = excluded.message,
                started_at = excluded.started_at,
                end_at = excluded.end_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (message, started_at.isoformat(), end_at_value),
        )
        await db.commit()


async def clear_maintenance_mode() -> None:
    async with aiosqlite.connect(logs_db_file) as db:
        await db.execute(
            """
            INSERT INTO maintenance_mode (id, active, message, started_at, end_at, updated_at)
            VALUES (1, 0, NULL, NULL, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT(id)
            DO UPDATE SET
                active = 0,
                message = NULL,
                started_at = NULL,
                end_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            """
        )
        await db.commit()


async def ensure_maintenance_access(uid: int | str) -> None:
    if int(uid) in await get_owner_ids():
        return

    status = await get_maintenance_status()
    if status.active and status.message is not None:
        raise MaintenanceModeActive(status.message, status.end_at)


@lightbulb.hook(
    lightbulb.ExecutionSteps.CHECKS,
    skip_when_failed=True,
    name="maintenance_check",
)
async def maintenance_check(_: lightbulb.ExecutionPipeline, ctx: lightbulb.Context):
    await ensure_maintenance_access(ctx.user.id)
