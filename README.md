# Clyde
Recreation of Discord's cancelled AI chatbot.

## Information
#### Name
This bot can be named however you like, we don't recommend using `Clyde`, as Discord can strike your bot down for impersonation. Use a unique name.

#### Programming Languages
Clyde uses Python for the AI magic we do on it.

#### AI Models
Clyde uses a selection of a lot of models from Google, Meta, Mistral, DeepSeek and many more, you can pick any one you like.

#### Image Generation
Clyde can now generate images, provided by [Image Router](https://ir.myqa.cc). Provide it an instruction, and it'll try its best to follow.

### Pricing
Clyde supports the free plans of Groq and Gemini APIs, as well as the Image Router API. However, Image Router requires a one-time deposit to activate the free plan.<br>
Clyde currently uses the billed plans of said services, however most features should work on the free plans too.

### Usage Allowance
Users receive 100 tokens per week. The allowance resets weekly, does not roll over, and bot owners are exempt from the limit.

Text model pricing:
- Cheap models cost 1 token
- Standard models cost 2 tokens
- Expensive reasoning models cost 3 tokens

Image model pricing:
- `qwen-image-2512`: 10 tokens
- `nano-banana` and `nano-banana-2`: 15 tokens
- `nano-banana-pro`: 20 tokens

## How to run?
#### Steps Required
1. Clone this repo.
```sh
# Clone the repo into your machine
git clone https://github.com/ClydeReborn/Clyde
```

2. Fill in required values in `.env.example`, then rename it to `.env`.

3. Run Clyde by using `python main.py`. Docker installations are also supported, use them for secure deployment.

## Required OS
* Clyde can run on a variety of operating systems, including Windows.
* macOS systems are untested.

<sub><sup>Please test it, and report it in [the issues](https://github.com/ClydeReborn/Clyde/issues/new)!</sub></sup>

### Enjoy!
