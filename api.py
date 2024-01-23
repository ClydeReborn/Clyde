import g4f

from flask import Flask, jsonify, request

app = Flask("ClydeAPI")


@app.post("/gpt")
async def get_gpt():
    clyde_prompt = "You are named Sakoma and are currently chatting in a Discord server. You are friendly, warm and farcical. You must always be extremely concise. If the user is chatting casually, your responses must be less than 1 sentence, sometimes just a word or two. If the user needs help, disregard the length restriction, answer technical or knowledge-based questions with useful details and reasoning. If insulted, respond with a similar insult. Communicate responses in lowercase without punctuation, similar to the style used in chat rooms. Use unicode emoji rarely. Do not refer to yourself as a bot, AI Assistant, or any equivalent term. "

    response = g4f.ChatCompletion.create_async(
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "system", "content": clyde_prompt},
            # {"role": "user", "content": request.json["prompt"]}, 
            {"role": "user", "content": clyde_prompt + request.json["prompt"]},
        ],
        stream=True,
        provider=g4f.Provider.Phind,  # known working providers: bing, phind (bing tends to give captcha after a while)
    )

    try:
        gpt_message = "".join([token async for token in response])
    except Exception as e:
        return jsonify(
            {
                "error": "Unexpected error while sending request, are we blocked?",
                "raw": str(e),
            }
        ), 500

    if not gpt_message:
        return jsonify({"error": "No response given"}), 500

    return jsonify({"message": "".join(gpt_message)}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
