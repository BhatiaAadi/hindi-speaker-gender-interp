#!/bin/bash
# Push code to turing (only tracked project folders, never personal files).
cd "$(dirname "$0")/.."
rsync -az --delete --exclude '__pycache__' --exclude 'results/' src scripts data requirements.txt turing:hindi_interp/
