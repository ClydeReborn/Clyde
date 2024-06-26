#!/bin/bash
set -e
shopt -s extglob

BOTNAME="Clyde"

if [ ! -d .venv ]; then
  echo ""
  echo "Configuring $BOTNAME..."
  git clone https://github.com/ClydeReborn/API
  mv API/!(README.md) .
  rm -rf API
  python -m venv .venv
  source .venv/bin/activate
  pip install -U -r requirements.txt
  echo ""
  read -p "Do you want to run a selfbot? (not recommended) [y/n] " choice
  if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo "Making a selfbot..."
      ln -sf clyde.selfbot.py clyde.py
  else
      echo "Continuing to make a bot..."
      ln -sf clyde.bot.py clyde.py
  fi
fi

if [ ! -f .env ]; then
  echo "Creating .env file..."
  echo 'TOKEN="<TOKEN_GOES_HERE>"' > .env
  echo "OWNER=<YOUR_DISCORD_ID_GOES_HERE>" >> .env
  echo "ERROR_CHANNEL=<ID_OF_ERROR_LOGGING_CHANNEL_GOES_HERE>" >> .env
  echo ""
  echo -e "\033[0;31mDon't forget to configure the .env file!\033[0m"
  echo ""
  exit
fi

source .venv/bin/activate
echo "$BOTNAME configured successfully!"
echo "Starting up $BOTNAME..."
pkill -O 60 -9 python
python -u mux.py
