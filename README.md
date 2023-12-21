# clyde/sakoma
 Recreation of Discord's canceled AI Bot (Clyde)

# story
So in March 2nd 2023, Discord introduced Clyde an AI ChatBot based on OpenAI (ChatGPT) however it had many flaws and all people hated it (mostly from NTTS (No Text To Speech) audience) and Discord killed Clyde bot on December 1st 2023 as it would cost discord a lot of money and so they decided to kill it, The EOL date was announced on November 5th 2023 and all my friends and everyone in the Chomu Pradise Club everyone panicked and was very sad (infact Bohdan threw his A30S and cracked the screen even more than it was), however Luna (lun4h) knew that this would happen and already started working on a backup.

# info
### name
the name for this bot is "Sakoma" suggested by 3177 
### programming language's
its using python for the bot and Shell for launching
### AI model
its using the Bing AI model based on GPT4 (better than clyde which used GPT3.5)

# Build
firstly clone this repo
```
git clone https://github.com/debarkak/clyde,git
```

then install python and required python modules

```
pip install -U g4f httpx
```

then edit the `clyde.py` file and modify this line
```
client.run("TOKEN_GOES_HERE")
```

in `"TOKEN_GOES_HERE"` replace this text with your token and remove the quotes (dont remove the brackets)

###
after that run `python test_clyde.py` to test the bot, if it works then run `chmod +x clyde.sh` then run `./clyde.sh` to run the bot. Now enjoy!

# note
* you MUST use a normal discord account, NOT a discord bot account
* currently you cannot run this bot in windows or other OS, only in Linux (recomended linux distros are Arch, Fedora and Debian)



