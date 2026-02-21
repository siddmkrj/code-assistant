# coco — CLI Coding Agent

A command-line coding agent powered by LangChain, LangGraph, and Anthropic Claude.

## Features

- **Multi-agent system**: specialized agents for code, planning, Q&A, and web search
- **Auto-routing**: automatically selects the best agent and model for each task
- **Codebase indexing**: semantic search over your codebase via Chroma + embeddings
- **Context compression**: auto-summarizes long conversations to save tokens
- **Human-in-the-loop**: asks for clarification when needed
- **LLM-agnostic**: built on LangChain abstractions, swap providers easily
- **Persistent caching**: SQLite cache for LLM responses
- **Session history**: full JSONL logs of all interactions

## Installation

```bash
pip install coco
```

Or from source:

```bash
git clone <repo>
cd code-assistant
pip install -e .
```

## Setup

1. Copy `.env.example` to `.env` and add your Anthropic API key:
   ```bash
   cp .env.example .env
   # Edit .env and set ANTHROPIC_API_KEY=...
   ```

2. Run `coco` in any directory:
   ```bash
   coco
   ```

3. (Optional) Run `/setup` inside coco to configure per-project settings.

## Usage

```
coco> /help              # Show all commands
coco> /index             # Index current codebase for semantic search
coco> /setup             # Configure models and settings
coco> /status            # Show current config and index stats

# Direct agent commands
coco> /ask how does this codebase work?
coco> /code add a type hint to all functions in utils.py
coco> /plan build a REST API with authentication
coco> /search latest LangGraph documentation

# Auto-routed (coco picks the best agent)
coco> What does the authenticate() function do?
coco> Fix the bug in my parser
coco> /clear             # Reset conversation
coco> /exit              # Exit
```

## Configuration

coco stores project config in a `.coco` file (TOML) in the current directory.
Global settings live in `~/.coco/config.toml`.

```toml
[model]
complex_model = "claude-sonnet-4-6"
simple_model = "claude-haiku-4-5-20251001"
temperature = 0.0
max_tokens = 4096

[index]
chunk_size = 1000
chunk_overlap = 200

[safety]
confirm_file_writes = true
confirm_shell_commands = true
```

## Architecture

```
coco/
├── config/      # TOML config management
├── memory/      # Context compression + JSONL history
├── tools/       # LangChain @tool definitions
├── indexer/     # Chroma codebase indexing
├── graph/       # LangGraph StateGraph workflow
├── agents/      # Specialized agent implementations
└── cli/         # Rich terminal UI + prompt_toolkit REPL
```

## Security

- API keys are read from environment variables only, never written to config files
- File writes require confirmation (configurable)
- Shell commands use argument lists (no shell injection)
- Add `.env` to your `.gitignore`
