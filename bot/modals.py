import asyncio
import uuid
from dataclasses import dataclass

import hikari
import miru
import lightbulb


@dataclass(frozen=True)
class InputModalResult:
    request: str
    context: miru.ModalContext


class InputModal(miru.Modal):
    def __init__(self, future: asyncio.Future[InputModalResult]) -> None:
        super().__init__(title="AI Request", custom_id=f"ai-input-{uuid.uuid4()}")
        self.future = future

        self.request_input = miru.TextInput(
            label="Your request",
            style=hikari.TextInputStyle.PARAGRAPH,
            required=True,
            max_length=4000,
            placeholder="Write your request here...",
        )
        self.add_item(self.request_input)

    async def callback(self, ctx: miru.ModalContext) -> None:
        if not self.future.done():
            self.future.set_result(InputModalResult(self.request_input.value, ctx))

        await ctx.defer(hikari.ResponseType.DEFERRED_MESSAGE_CREATE)

    async def on_error(
        self, error: Exception, context: miru.ModalContext | None = None
    ) -> None:
        if not self.future.done():
            self.future.set_exception(error)
        if context is not None:
            await context.respond(
                "There was an internal error handling your request.", flags=64
            )


@dataclass(frozen=True)
class MaintenanceModalResult:
    reason: str
    context: miru.ModalContext


class MaintenanceModal(miru.Modal):
    def __init__(self, future: asyncio.Future[MaintenanceModalResult]) -> None:
        super().__init__(
            title="Start Maintenance", custom_id=f"maintenance-{uuid.uuid4()}"
        )
        self.future = future

        self.reason_input = miru.TextInput(
            label="Maintenance message",
            style=hikari.TextInputStyle.PARAGRAPH,
            required=True,
            max_length=1000,
            placeholder="Explain why maintenance is active and what users should know...",
        )
        self.add_item(self.reason_input)

    async def callback(self, ctx: miru.ModalContext) -> None:
        if not self.future.done():
            self.future.set_result(
                MaintenanceModalResult(reason=self.reason_input.value, context=ctx)
            )

        await ctx.defer(hikari.ResponseType.DEFERRED_MESSAGE_CREATE)

    async def on_error(
        self, error: Exception, context: miru.ModalContext | None = None
    ) -> None:
        if not self.future.done():
            self.future.set_exception(error)
        if context is not None:
            await context.respond(
                "There was an internal error handling maintenance mode.", flags=64
            )


async def prompt_with_modal(
    ctx: lightbulb.Context, inter_client: miru.Client, timeout: float = 300.0
) -> InputModalResult | None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[InputModalResult] = loop.create_future()

    modal = InputModal(future)
    await ctx.interaction.create_modal_response(
        modal.title,
        modal.custom_id,
        components=modal.build(),
    )
    inter_client.start_modal(modal)

    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        return None


async def prompt_maintenance_modal(
    ctx: lightbulb.Context, inter_client: miru.Client, timeout: float = 300.0
) -> MaintenanceModalResult | None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[MaintenanceModalResult] = loop.create_future()

    modal = MaintenanceModal(future)
    await ctx.interaction.create_modal_response(
        modal.title,
        modal.custom_id,
        components=modal.build(),
    )
    inter_client.start_modal(modal)

    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        return None
