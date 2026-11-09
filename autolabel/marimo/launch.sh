#!/bin/bash
# Starts the Codex session for the first-read notebook on the local remote-control daemon (gpt-6-astra, xhigh). Run inside tmux.
export PATH="$HOME/.config/nvm/versions/node/v24.18.0/bin:$HOME/.local/bin:$PATH"
cd "$HOME/dev/interp" || exit 1
exec codex --remote unix:// -m gpt-6-astra -c 'model_reasoning_effort="xhigh"' -C "$HOME/dev/interp" \
  "$(cat "$HOME/dev/interp/autolabel/marimo/PROMPT.md")"
