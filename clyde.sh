#!/bin/bash
echo "Starting up Clyde..."

source clyde-venv/bin/activate
nohup python api.py &
python clyde.py
