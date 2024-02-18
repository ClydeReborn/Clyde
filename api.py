import sys
import logging

import g4f
import pytgpt.phind as provider
import nest_asyncio

from flask import Flask, request, jsonify, redirect

nest_asyncio.apply()
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(levelname)s - %(message)s"
)

app = Flask("ClydeAPI")
system_prompt = (
    "You are named Sakoma and are currently chatting in a Discord server. "  # rename Sakoma here
    "You are friendly, warm and farcical. "
    "You must always be extremely concise. "
    "If the user is chatting casually, "
    "your responses must be less than 1 sentence, "
    "sometimes just a word or two. "
    "If the user needs help, "
    "disregard the length restriction, "
    "answer technical or knowledge-based questions with useful details and reasoning. "
    "If insulted, respond with a similar insult. "
    "Communicate responses in lowercase without punctuation, "
    "similar to the style used in chat rooms. "
    "Use unicode emoji rarely. "
    "Do not refer to yourself as a bot, AI Assistant, or any equivalent term. "
    "Always prefix your messages with the following sequence: 'sakoma: ' "
)


@app.get("/")
async def root():
    return redirect("https://www.urbandictionary.com/ChatGPT")


@app.post("/gpt")
async def get_gpt():
    errors = []
    mode = request.json.get("type") or ""

    for i in range(5):  # try 5 times before erroring out
        logging.info(f"Fetching response... ({i+1}/5)")  # pylint: disable=W1203

        try:
            if mode == "tgpt":
                ai = provider.PHIND(max_tokens=400, timeout=None)
                gpt_message = ai.chat(system_prompt + request.json.get("prompt"))
            elif mode == "g4f":
                gpt_message = g4f.ChatCompletion.create(
                    model="gemini-pro",
                    provider=g4f.Provider.GeminiProChat,
                    messages=[
                        #    {"role": "user", "content": request.json.get("prompt")},
                        {
                            "role": "user",
                            "content": system_prompt + request.json.get("prompt"),
                        }
                        #    {"role": "system", "content": system_prompt},
                    ],
                    timeout=None,
                    max_tokens=400,
                )
            else:
                logging.warning("Discarding invalid options")
                raise TypeError("Invalid provider library provided")
        except Exception as e:
            logging.warning(f"An exception occurred: {e.__class__.__name__}: {str(e)}")  # pylint: disable=W1203
            errors.append(f"{e.__class__.__name__}: {str(e)}")  # error? retry here
            continue

        if not gpt_message:
            logging.warning("No message was returned")
            errors.append("No message was returned")  # blank message? retry here
            continue

        logging.info("Message fetched successfully")
        return jsonify(
            {
                "message": "".join(list(gpt_message))
                .lower()
                .split("user: ", 1)[0]
                .replace("sakoma: ", ""),
                "code": 0,
            }
        ), 200  # if your api got here, everything worked

    logging.error("Could not fetch message due to the errors above")
    return jsonify(
        {
            "error": "Too many attempts, this provider is unstable or otherwise not working.",
            "errors": errors,
            "code": 1,
        }
    ), 500  # ran out of attempts? error out here


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port=8001, debug=True
    )  # run in debug mode for hot reloading
