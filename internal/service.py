import io
import os
import re
import time
from typing import Optional

import hikari
import lightbulb

import httpx
import groq
from google import genai
from google.genai import types
from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .handlers import exponential
from .config import (
    MAX_CHAT_HISTORY,
    IMAGE_MODELS,
    SAFETY_PROMPT,
    logger,
    chat_histories,
)


@lightbulb.di.with_di
async def get_owner_ids(bot: hikari.GatewayBot):
    owner_ids = []

    application = await bot.rest.fetch_application()
    if application.owner is not None:
        owner_ids.append(application.owner.id)
    if application.team is not None:
        owner_ids.extend(application.team.members.keys())

    return owner_ids


class AIService:
    """Service for handling AI-related operations"""

    @staticmethod
    def format_chat_history(messages: list[str]) -> list[dict]:
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
    @lightbulb.di.with_di
    async def check_llamaguard(
        text: str, groq_client: groq.AsyncGroq
    ) -> tuple[bool, int]:
        """Validate safety of a request or response"""

        logger.info("Checking safety")
        request = await groq_client.chat.completions.create(
            model="openai/gpt-oss-safeguard-20b",
            messages=[
                ChatCompletionSystemMessageParam(role="system", content=SAFETY_PROMPT),
                ChatCompletionUserMessageParam(role="user", content=text),
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
        is_safe = tag == "safe"

        logger.info(f"Verdict: {tag}, {code}")

        if tag == "unsafe" and code is None:
            return False, 2

        return is_safe, severity

    @staticmethod
    @exponential(3, 5, 30)
    @lightbulb.di.with_di
    async def generate_text_with_gemini(
        request: str,
        model: str,
        system_prompt: str,
        user_id: int,
        image: Optional[io.BytesIO],
        gemini_client: genai.Client,
    ) -> Optional[str]:
        """Generate text using Gemini models with native chat history"""

        is_safe, severity = await AIService.check_llamaguard(request)
        owner_ids = await get_owner_ids()
        if not is_safe and user_id not in owner_ids:
            return "I'm sorry, but I'm unable to assist with that."

        chat_histories.setdefault(user_id, [])

        config = types.GenerateContentConfig(system_instruction=system_prompt)

        history = []
        for i, msg in enumerate(chat_histories[user_id]):
            role = "user" if i % 2 == 0 else "model"
            history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg)])
            )

        if image:
            parts = [
                types.Part.from_text(text=request),
                types.Part.from_bytes(data=image.read(), mime_type="image/png"),
            ]
        else:
            parts = [types.Part.from_text(text=request)]

        history.append(types.Content(role="user", parts=parts))

        start_time = time.perf_counter()
        ttft = None
        full_text = []

        if "gemma" in model.lower():
            stream = await gemini_client.aio.models.generate_content_stream(
                model=model, contents=history
            )
        else:
            stream = await gemini_client.aio.models.generate_content_stream(
                model=model, config=config, contents=history
            )

        async for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - start_time
                logger.info(f"TTFT: {ttft:.3f}s")
            if chunk.text:
                full_text.append(chunk.text)

        end_time = time.perf_counter()
        raw_result = result = "".join(full_text)

        usage = chunk.usage_metadata
        if usage and (end_time - (start_time + ttft)) > 0:
            tps = usage.candidates_token_count / (end_time - (start_time + ttft))
            logger.info(f"TPS: {tps:.2f}")

        is_safe, severity = await AIService.check_llamaguard(raw_result)
        if not is_safe and user_id not in owner_ids:
            result = "Sorry, that's beyond my current scope."

        chat_histories[user_id].append(request)
        chat_histories[user_id].append(result)

        return result

    @staticmethod
    @exponential(3, 5, 30)
    @lightbulb.di.with_di
    async def generate_text_with_groq(
        request: str,
        model: str,
        system_prompt: str,
        user_id: int,
        groq_client: groq.AsyncGroq,
    ) -> Optional[str]:
        """Generate text using Groq models"""
        is_safe, severity = await AIService.check_llamaguard(request)
        owner_ids = await get_owner_ids()
        if not is_safe and user_id not in owner_ids:
            return "I'm sorry, but I'm unable to assist with that."

        chat_histories.setdefault(user_id, []).append(request)
        formatted_history = AIService.format_chat_history(chat_histories[user_id])

        start_time = time.perf_counter()
        ttft = None
        full_text = ""

        response = await groq_client.chat.completions.create(
            model=model,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=system_prompt,
                ),
                *formatted_history,
            ],
            stream=True,
        )

        async for chunk in response:
            if ttft is None and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - start_time
                logger.info(f"TTFT: {ttft:.3f}s")

            if len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content or ""
                full_text += content

            if chunk.usage is not None:
                total_tokens = chunk.usage.completion_tokens
                total_time = time.perf_counter() - start_time
                generation_time = total_time - ttft
                tps = total_tokens / generation_time if generation_time > 0 else 0
                logger.info(f"TPS: {tps:.2f}")

        result = full_text

        is_safe, severity = await AIService.check_llamaguard(result)
        if not is_safe and user_id not in owner_ids:
            result = "Sorry, that's beyond my current scope."

        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)

        chat_histories[user_id].append(result)
        return result

    @staticmethod
    @lightbulb.di.with_di
    async def generate_image(
        model: str,
        prompt: str,
        user_id: str,
        gemini_client: genai.Client,
    ) -> Optional[io.BytesIO | str]:
        """Generate an image from text prompt"""
        is_safe, severity = await AIService.check_llamaguard(prompt)
        owner_ids = await get_owner_ids()
        if not is_safe and user_id not in owner_ids:
            return "I'm sorry, but I cannot generate this image."

        if model == "qwen-image-2512":
            async with httpx.AsyncClient(timeout=60.0) as img_client:
                response = await img_client.post(
                    "https://ir-api.myqa.cc/v1/openai/images/generations",
                    json={
                        "model": "qwen/qwen-image-2512",
                        "prompt": prompt,
                    },
                    headers={
                        "Authorization": f"Bearer {os.getenv('IMAGE_GEN_TOKEN')}",
                        "Content-Type": "application/json",
                    },
                )

                response.raise_for_status()
                data = response.json()

                if not data.get("data"):
                    failure_reason = data["error"]["message"]

                    if "safety" in failure_reason:
                        return "The request was filtered by the provider. Try again using a different prompt."
                    else:
                        return f"The request failed because `{failure_reason}`. Try again using a different prompt."

                image_data = await img_client.get(data["data"][0]["url"])
                image = io.BytesIO(image_data.content)
                image.seek(0)

            return image
        elif model in IMAGE_MODELS.keys():
            response = await gemini_client.aio.models.generate_content(
                model=IMAGE_MODELS.get(model),
                contents=[prompt],
            )

            image = None
            for part in response.parts:
                if part.inline_data:
                    raw_image = part.as_image()
                    image = io.BytesIO(raw_image.image_bytes)
                    image.seek(0)

                    break  # if Nano Banana returns several images, this shouldn't happen

            if not image:
                raise RuntimeError("no data received")

            return image
        else:
            raise ValueError(f"model {model} does not generate images")


async def generate_text(
    request: str,
    model: str,
    prompt: str,
    user_id: int,
    image: Optional[io.BytesIO] = None,
) -> Optional[str]:
    """Unified text generation function that routes to appropriate service"""
    try:
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
