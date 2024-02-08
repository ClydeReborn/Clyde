#!/bin/bash
echo "Starting up Clyde..."

deactivate  # comment this and the line below if you don't use a venv
# shellcheck disable=SC1091
source /root/runner/public/clyde/clyde-venv/bin/activate
nohup python api.py &
nohup python clyde.py &
wait
