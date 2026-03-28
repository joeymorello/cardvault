#!/bin/bash
cd /home/sprite/gt/cardvault/backend
exec /.sprite/languages/python/pyenv/versions/3.13.7/bin/python3 -m uvicorn server:app --host 0.0.0.0 --port 8080
