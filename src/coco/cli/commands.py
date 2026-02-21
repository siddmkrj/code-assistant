"""Command parsing and registry for coco.

Commands start with '/'. Everything else is a natural-language prompt
that gets auto-routed to the best agent.
"""
from __future__ import annotations

# Maps command name → handler method name on CocoApp
COMMANDS: dict[str, str] = {
    "help":   "cmd_help",
    "index":  "cmd_index",
    "setup":  "cmd_setup",
    "exit":   "cmd_exit",
    "quit":   "cmd_exit",
    "ask":    "cmd_ask",
    "code":   "cmd_code",
    "plan":   "cmd_plan",
    "search": "cmd_search",
    "clear":  "cmd_clear",
    "status": "cmd_status",
    "history": "cmd_history",
}

HELP_TEXT = """
## coco — Command Reference

### Automatic Routing
Just type naturally — coco will pick the best agent automatically.

### Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/index` | Index the current codebase for semantic search |
| `/setup` | Configure models and settings (saves to `.coco`) |
| `/status` | Show current configuration and index stats |
| `/clear` | Clear conversation history and start fresh |
| `/history` | Show path to session history log |
| `/exit`, `/quit` | Exit coco |

### Direct Agent Commands
These bypass auto-routing and go directly to a specific agent:

| Command | Agent | Model |
|---------|-------|-------|
| `/ask <question>` | Ask agent | Haiku (fast) |
| `/code <task>` | Code agent | Sonnet (smart) |
| `/plan <task>` | Plan agent | Sonnet (smart) |
| `/search <query>` | Search agent | Haiku + DuckDuckGo |

### Tips
- Run `/index` first to enable semantic codebase search
- Use `/plan` before `/code` for complex tasks
- Use `/setup` to configure per-project model preferences
- History is saved to `~/.coco/history/` as JSONL
"""


def parse_command(user_input: str) -> tuple[str | None, str]:
    """Parse user input into (command_name, args).

    Returns:
        (command_name, args) if input starts with '/'
        (None, original_input) if it's a natural-language prompt
    """
    stripped = user_input.strip()
    if not stripped.startswith("/"):
        return None, stripped

    # Handle bare "/" or "//"
    content = stripped[1:].strip()
    if not content:
        return None, stripped

    parts = content.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return cmd, args
