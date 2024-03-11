# Clyde
A recreation of Discord's cancelled AI chatbot: Clyde.

## Information
#### Name
The name for this bot should be Clyde, but you can rename it.<br>
The pre-deployed bot's name was Sakoma, we moved a to proper Clyde bot account.

#### Programming Languages
Clyde uses Python for literally everything except spawning in the subprocesses, that's done by Bash.

#### AI Models
Clyde tries to use the best providers from [python-tgpt](https://github.com/Simatwa/python-tgpt) and [gpt4free](https://github.com/xtekky/gpt4free), these are subject to change depending on what providers are online.

Currently we use these:
- [Phind@python-tgpt](https://github.com/Simatwa/python-tgpt/blob/main/src/pytgpt/phind/main.py): GPT-3.5
- [FlowGpt@g4f](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/FlowGpt.py): Gemini Pro

We would recommend to use one of the following providers for your instance of Clyde:
- gpt4free: Bing, FlowGpt, Llama2, Phind
- python-tgpt: Llama2, Phind

<sub><sup>Bing may give CAPTCHAs after a while or when providing the wrong values.</sub></sup><br>
<sub><sup>Phind (TGPT) has the tendency to say something unrelated if it didn't understand your prompt.</sub></sup><br>
<sub><sup>Phind (G4F) may give errors or blank responses, use the `python-tgpt` implementation.</sub></sup><br>
<sub><sup>Llama2 may work abnormally slow or not return any responses.</sub></sup>
<sub><sup>FlowGpt can be easily ratelimited.</sub></sup>

## How to run?
#### Steps Required
1. Clone this repo.
```sh
# Clone the repo into your machine
git clone https://github.com/ClydeReborn/Clyde
```

2. Run `./clyde.sh` to immediately configure a copy of Clyde.<br>
<sub><sup>Be careful when running Clyde as a selfbot, as it violates Discord ToS if you do so.</sub></sup>

3. Fill in your bot or user token, your user ID and an error logging channel the in `.env` file.

4. Run `./clyde.sh` again to run Clyde.
#### Testing Providers
To check working providers, check out the [Tests](https://github.com/ClydeReborn/Tests).<br>
The comprehensive test may take a longer time to finish!

## Required OS
* Clyde can only run on Linux. We recommend using Arch Linux, Fedora or Debian for this.<br>
If you're using Windows, you may be able to run Clyde on WSL.

Enjoy!
