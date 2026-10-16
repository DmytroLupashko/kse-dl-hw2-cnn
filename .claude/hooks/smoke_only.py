"""PreToolUse hook: block notebook execution unless it is a smoke run (SMOKE=1) that writes no outputs."""

import json
import sys

command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
executes = ("nbconvert" in command and "--execute" in command) or "papermill" in command
if executes and "SMOKE=1" not in command:
    print("Full notebook runs are the user's. Prefix the command with SMOKE=1.", file=sys.stderr)
    sys.exit(2)
if executes and "--stdout" not in command:
    print("Smoke runs must not write outputs. Use --stdout > /dev/null.", file=sys.stderr)
    sys.exit(2)
