# Clyde
Recreation of Discord's cancelled AI chatbot.

## Information
#### Name
The name for this bot should be Clyde, but you can rename it.<br>
The pre-deployed bot's name was Sakoma, we moved a to proper Clyde bot account.

#### Programming Languages
Clyde uses Python for literally everything except spawning in the subprocesses, that's done by Bash.

#### AI Models
Clyde tries to use any working provider from either [python-tgpt](https://github.com/Simatwa/python-tgpt) or [gpt4free](https://github.com/xtekky/gpt4free), the current provider will change over time.

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
<sub><sup>Don't use Ubuntu!</sub></sup>
* If you're using Windows, you may be able to run Clyde on WSL.<br>
<sub><sup>Don't let Microsoft bloat it up with ads!</sub></sup>
* We are not sure if this will run on macOS/BSD as it has not been tested.<br>
<sub><sup>Someone test it, and report it in [the issues](https://github.com/ClydeReborn/Clyde/issues/new)!</sub></sup>

### Enjoy!
