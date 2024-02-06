#!/bin/zsh
echo "Starting up Clyde..."

deactivate  # comment these lines if you dont use a venv <-----
source /root/runner/public/clyde/clyde-venv/bin/activate # <---
python api.py &!
python clyde.py
