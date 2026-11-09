#!/bin/bash
# Launches the AUTOLABEL-CAMMARATA Claude Code session (Fable, remote control on). Run inside tmux.
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/dev/interp" || exit 1
exec claude --model fable --effort xhigh --permission-mode auto \
  --name AUTOLABEL-CAMMARATA --remote-control AUTOLABEL-CAMMARATA \
  "$(cat "$HOME/dev/interp/autolabel/LAUNCH_PROMPT.md")"
