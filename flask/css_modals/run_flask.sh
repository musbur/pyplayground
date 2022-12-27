#!/bin/sh

export FLASK_APP=app
export FLASK_DEBUG=1
export FLASK_RUN_PORT=8008
export FLASK_RUN_HOST=0.0.0.0

#python -m flask shell
python -m flask run

