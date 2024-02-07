from g4f.Provider import __all__, ProviderUtils
from g4f import ChatCompletion
import concurrent.futures

_ = ["BaseProvider", "AsyncProvider", "AsyncGeneratorProvider", "RetryProvider"]
system_prompt = "You are named Sakoma and are currently chatting in a Discord server. You are friendly, warm and farcical. You must always be extremely concise. If the user is chatting casually, your responses must be less than 1 sentence, sometimes just a word or two. If the user needs help, disregard the length restriction, answer technical or knowledge-based questions with useful details and reasoning. If insulted, respond with a similar insult. Communicate responses in lowercase without punctuation, similar to the style used in chat rooms. Use unicode emoji rarely. Do not refer to yourself as a bot, AI Assistant, or any equivalent term."


def test_provider(provider):
    try:
        provider = ProviderUtils.convert[provider]
        if provider.working and not provider.needs_auth:
            print(f"\033[95mTesting {provider.__name__} without a system prompt")
            completion = ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": system_prompt + "hi who are you"}
                ],
                provider=provider,
            )
            return completion, provider.__name__
    except Exception:
        print(f"\033[91m{provider.__name__} isn't working.")
        return None


def system_test(provider):
    try:
        provider = ProviderUtils.convert[provider]
        if provider.working and not provider.needs_auth:
            print(f"\033[95mTesting {provider.__name__} with a system prompt")
            completion = ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "hi who are you"},
                ],
                provider=provider,
            )
            return completion, provider.__name__
    except Exception:
        print(f"\033[91m{provider.__name__} isn't accepting system prompts.")
        return None


with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(test_provider, provider)
        for provider in __all__
        if provider not in _
    ]

    futures_system = [
        executor.submit(system_test, provider)
        for provider in __all__
        if provider not in _
    ]

    for future in concurrent.futures.as_completed(futures):
        if result := future.result():
            print(f"\033[92m{result[1]} is working: {result[0]}")

    for future in concurrent.futures.as_completed(futures_system):
        if result := future.result():
            print(f"\033[92m{result[1]} accepted the system request: {result[0]}")
