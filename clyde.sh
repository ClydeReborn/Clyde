#!/bin/bash
if [ ! -d .venv ]; then
  echo "Configuring Clyde..."
  git clone https://github.com/ClydeReborn/API
  mv API/* .
  rm -rf API
  python -m venv .venv
  pip install -U -r requirements.txt
  ln -sf clyde.bot.py clyde.py
fi

source .venv/bin/activate
 
echo "Starting up Clyde..."
python mux.py
