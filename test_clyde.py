import g4f
import time


def test(system: bool, provider: g4f.Provider) -> bool:
    clyde_prompt = "You are named Clyde and are currently chatting in a Discord server. You are friendly, warm and farcical. You must always be extremely concise. If the user is chatting casually, your responses must be less than 1 sentence, sometimes just a word or two. If the user needs help, disregard the length restriction, answer technical or knowledge-based questions with useful details and reasoning. If insulted, respond with a similar insult. Communicate responses in lowercase without punctuation, similar to the style used in chat rooms. Use unicode emoji rarely. Do not refer to yourself as a bot, AI Assistant, or any equivalent term. "
    clyde_prompt = clyde_prompt.lower()

    if system:
        gpt_message = g4f.ChatCompletion.create(
            provider=provider,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "i like you"},
                      {"role": "system", "content": clyde_prompt}],
            stream=True,
            timeout=10,
        )
    else:
        gpt_message = g4f.ChatCompletion.create(
            provider=provider,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": clyde_prompt + "i like you"}],
            stream=True,
            timeout=10,
        )
    try:    
        full_message = "".join([token for token in gpt_message])
        alpha = list(filter(str.isalpha, full_message))
        ratio = sum(map(str.islower, alpha)) / len(alpha)
        if full_message != "":
            print(f"Response: {full_message} ({round(ratio*100, 2)}% lowercase)")
    except Exception as e:
        newline = "\n"
        print(f"FAILED: {e.__class__.__name__}: {str(e).split(newline)[0]}")
        return False
    
    if full_message == "":
        print("FAILED: no response")
        return False

    if ratio <= 0.98:
        print("FAILED: lowercase ratio isn't >=98%")
        return False
    
    print("SUCCESS: all checks passed")
    return True

def gather_tests(provider: g4f.Provider) -> tuple[bool, int, int]:
    successes = 0
    failures = 0

    for _ in range(10):
        time.sleep(2)
        result = test(True, provider)
        if result:
            successes += 1
        else:
            failures += 1

    if failures:
        print(f"NO: Provider {provider.__name__} unsuitable for Clyde")
        return (False, successes, failures)
    print(f"YES: Provider {provider.__name__} suitable for Clyde")
    return (True, successes, failures)

providers = [
    provider
    for provider in g4f.Provider.__providers__
    if provider.working and not provider.needs_auth and provider.supports_stream
]

for provider in providers:
    print(f"Testing {provider.__name__}")
    results = gather_tests(provider)
    print(f"Success: {results[1]}, Failed: {results[2]}\n\n\n\n")
