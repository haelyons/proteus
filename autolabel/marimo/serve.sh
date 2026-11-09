#!/bin/bash
# Serves the first-read notebooks in app mode (code hidden, not editable) on 127.0.0.1:2731 with a token password:
# InceptionV1 (autolabel/marimo/first_read.py) and ESM-2 (autolabel/esm/first_read_esm.py); the root is a gallery.
# Exposed publicly by: tailscale funnel --bg --https=8443 http://127.0.0.1:2731   (stop: tailscale funnel --https=8443 off)
cd "$HOME/dev/interp" || exit 1
exec janus/.venv-gpu/bin/marimo run autolabel/marimo/first_read.py autolabel/esm/first_read_esm.py \
  --host 127.0.0.1 --port 2731 --headless --token-password-file "$HOME/.config/autolabel-marimo/token"
