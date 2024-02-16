#!/bin/bash
echo "Starting up Clyde..."

source clyde-venv/bin/activate
python api.py &
python clyde.py
