#!/bin/zsh
echo "Starting up Clyde..."
python api.py &!
python clyde.py
