# Clyde
A recreation of Discord's cancelled AI chatbot: Clyde.

## Story
On March 2nd 2023, Discord introduced Clyde, an AI chatbot based on OpenAI's ChatGPT, however it had many flaws, and all people hated it, mostly from NTTS' (No Text To Speech) audience, and Discord killed Clyde on December 1st 2023, as it would cost Discord a lot of money and so they decided to shut it down, the EOL date was announced on November 5th 2023 and all of debarkak's friends and everyone in the Chomu Paradise Club panicked and was very sad (infact Bohdan threw his A30s and cracked the screen even more than it was), however Luna knew that this would happen, and already started working on a backup.

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
- [Aura@g4f](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/Aura.py): GPT-3.5

We would recommend to use one of the following providers for your instance of Clyde:
- gpt4free: Bing¹, Aura, Llama2⁵, Phind⁴
- python-tgpt: Llama2⁵, Phind³

<sub><sup>¹May give CAPTCHAs after a while or when providing the wrong values.</sub></sup><br>
<sub><sup>²If using the vendored provider file; check the [Providers](https://github.com/ClydeReborn/Providers).</sub></sup><br>
<sub><sup>³Has the tendency to say something unrelated if it didn't understand your prompt.</sub></sup><br>
<sub><sup>⁴May give errors or blank responses, use the `python-tgpt` implementation.</sub></sup><br>
<sub><sup>⁵May work abnormally slow or not return any responses.</sub></sup>

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
