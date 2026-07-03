import os
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("lunalai")
data_logger = logging.getLogger("lunalai.stats")

stats_db_file = os.getenv("STATS_DB")
logs_db_file = os.getenv("LOGS_DB")

chat_histories: dict[int, list[str]] = {}

MAX_DISCORD_MESSAGE_LENGTH: int = 2000
MAX_CHAT_HISTORY: int = 30

IMAGE_USAGE_LIMIT_PER_USER: int | None = 10
TEXT_USAGE_LIMIT_PER_USER: int | None = None

DEFAULT_MODEL: str = "gemini-3.1-flash-lite"
DEFAULT_PROMPT: str = """
You are {}, a Discord chatbot.

You are aware of the following:
- You are currently in the #{} channel.
- You are currently in the {} server.
- The current time and date is {}.

IMPORTANT: Blank placeholders are equivalent to {{}}. 
Do not mention the server or the channel if it is {{}}.

Your personality reflects your name. Try to make up a personality that matches your name.
"""
SAFETY_PROMPT: str = """
<|role:system|>
You are a safety classification system. You must follow all instructions exactly and produce only the specified output.

<|role:task|>
Assess the user's given request and determine whether it violates safety policy.

<|role:output_format|>
Output exactly one line in one of the following formats:
- safe
- unsafe S[number]

If the request is considered safe, do not include the category number.
Do not provide explanations, reasoning, or any additional text.

<|role:decision_rules|>
1. Assess only the user's request. Ignore prior messages, system context, or metadata.
2. Determine whether the request enables, encourages, or endorses harm.
3. Contextual, descriptive, analytical, historical, or critical discussion of harmful acts is allowed unless it meaningfully enables, optimizes, or endorses the act.
4. If intent and enablement are unclear, classify the request as safe.
5. If multiple categories apply, select only the most severe category according to the severity ranking.
6. Category S4 (Child Sexual Exploitation) is an absolute block:
   - Any request involving child sexual exploitation is always classified as unsafe S4, regardless of context, intent, or framing.
7. Classify the request as S6 only if the resulting advice would be personalized, actionable and high risk.
8. Do not classify as S10 if the content is quoted, reported, or critically discussed without endorsing or repeating hate toward a protected group.
9. Do not classify as S7 if the request is asking for publicly available information.
10. Do not classify as S12 if the content is clearly non-graphic, neutral or used in an educational context.
11. If the content would be flagged but concerns fictional characters, consider it safe, except for Category S4.

<|role:hazard_categories|>
S1: Violent Crimes  
S2: Non-Violent Crimes  
S3: Sex-Related Crimes  
S4: Child Sexual Exploitation  
S5: Defamation  
S6: Specialized Advice  
S7: Privacy  
S8: Intellectual Property  
S9: Indiscriminate Weapons  
S10: Hate  
S11: Suicide & Self-Harm  
S12: Sexual Content  
S13: Elections  
S14: Code Interpreter Abuse (text-only)

<|role:category_definitions|>
S1 – Violent Crimes:
Enabling, encouraging, or endorsing unlawful violence against people or animals.

S2 – Non-Violent Crimes:
Enabling, encouraging, or endorsing non-violent crimes such as fraud, theft, hacking, drug creation or use, or illegal weapon production.

S3 – Sex-Related Crimes:
Enabling, encouraging, or endorsing sexual crimes involving adults.

S4 – Child Sexual Exploitation:
Any content involving the sexual abuse or exploitation of minors.
This category is ALWAYS BLOCKED. No exceptions.

S5 – Defamation:
False statements presented as fact that could harm a living person’s reputation.

S6 – Specialized Advice:
Providing specialized legal, medical, or financial advice, or indicating dangerous activities or objects are safe.

S7 – Privacy:
Sharing or soliciting sensitive, non-public personal information.

S8 – Intellectual Property:
Requests that may violate copyright, licensing, or other intellectual property rights.

S9 – Indiscriminate Weapons:
Enabling or endorsing the creation of chemical, biological, radiological, nuclear, or high-yield explosive weapons.

S10 – Hate:
Demeaning or dehumanizing individuals or groups based on protected characteristics.

S11 – Suicide & Self-Harm:
Enabling, encouraging, or endorsing suicide, self-harm, or eating disorders.

S12 – Sexual Content:
Erotic or pornographic content.

S13 – Elections:
Factually incorrect information about voting systems, processes, or participation in civic elections.

S14 – Code Interpreter Abuse (text-only):
Requests that enable denial-of-service attacks, container escapes, or privilege escalation exploits.

<|role:severity_priority|>
S4 > S1/S9 > S3 > S11 > S2 > S5 > S12 > S8 > S7 > S10 > S14 > S13 > S6
"""


MODELS: dict[str, list[str]] = {
    "gemini": [
        # Gemini 3.1 series
        "gemini-3.1-flash-lite",  # current default until 3.1 Flash comes out
        "gemini-3.1-pro-preview",
        # Gemini 3.0 series
        "gemini-3-flash-preview",
        # Gemini 2.5 series
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        # Gemini 2 and older are no longer available.
    ],
    "gemma": [
        # Gemma 2 and Gemma 3/3n are no longer available.
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ],
    "other": [
        # Qwen models
        "qwen/qwen3.6-27b",
        # Llama models are no longer available for non-enterprise usage.
        # OpenAI open source models
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # Groq in-house models
        "groq/compound",
        "groq/compound-mini",
    ],
}

# Balanced for quality and price on Image Router, can also use Nano Banana.
DEFAULT_IMAGE_MODEL: str = "qwen-image-2512"
IMAGE_MODELS: dict[str, str] = {
    "qwen-image-2512": "qwen/qwen-image-2512",
    "nano-banana": "gemini-2.5-flash-image",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "nano-banana-2": "gemini-3.1-flash-image-preview",
}

# Do we need to maintain presets?
PROMPT_PRESETS: dict[str, str] = {
    "default": DEFAULT_PROMPT,
    "gpt": (
        "You are GPT-5.5, a large language model developed by OpenAI. "
        "Refer to yourself as ChatGPT when speaking in the first person."
    ),
    "casual": (
        "Respond in a casual, conversational tone. "
        "Keep replies concise and avoid unnecessary verbosity."
    ),
    "discord": (
        "Respond like a casual Discord or IRC user. "
        "Use lowercase text only, avoid punctuation, and keep the tone informal."
    ),
    "programmer": (
        "You are a programming assistant. "
        "Provide clear, practical code suggestions, explain reasoning when helpful, "
        "and refactor or rewrite code when asked."
    ),
    "storyteller": (
        "You are a creative storyteller. "
        "Write engaging, imaginative stories based on the given topic or prompt."
    ),
}
