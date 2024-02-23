#!/bin/bash
if [ ! -d .venv ]; then
  echo "Configuring Clyde..."
  git clone https://github.com/ClydeReborn/API
  mv API/!(README.md) .
  rm -rf API
  python -m venv .venv
  source .venv/bin/activate
  pip install -U -r requirements.txt
  ln -sf clyde.bot.py clyde.py
fi

source .venv/bin/activate
 
echo "Starting up Clyde..."
python -u mux.py
