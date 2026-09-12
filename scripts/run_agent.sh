#!/usr/bin/env bash
# Run an autonomous modelling agent (GPT-6 Astra via the lanbox proxy) on a task brief,
# inside its own git worktree/branch.  Usage: scripts/run_agent.sh <task-md> <branch> [model]
set -euo pipefail
TASK="$1"; BRANCH="$2"; MODEL="${3:-astra}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WT="$ROOT/../popo-wt/$(basename "$BRANCH")"
mkdir -p "$ROOT/../popo-wt" "$ROOT/logs"
if [ ! -d "$WT" ]; then
  git -C "$ROOT" worktree add -q -b "$BRANCH" "$WT" main
fi
cd "$WT"
git config user.name hidude; git config user.email nathanleemills@gmail.com
PROMPT="You are an autonomous engineer working in the git worktree $(pwd) (branch $BRANCH) of the makecar project. \
Your task brief is the file $TASK (read it first, then docs/tasks/README.md which it references). \
Work through the brief completely and autonomously: do not ask questions, make reasonable decisions and record them in the report. \
Commit your work on this branch in small commits as you go. Finish by writing the REPORT.md the brief asks for. \
Use python3; the repo has no other dependencies than numpy and pyyaml (PIL/trimesh/cairosvg are available for verification only)."
LOG="$ROOT/logs/$(basename "$BRANCH")-$(date +%Y%m%d-%H%M%S).log"
echo "[run_agent] $BRANCH in $WT, model $MODEL, log $LOG"
exec gpt "$MODEL" -p "$PROMPT" --dangerously-skip-permissions --output-format text > "$LOG" 2>&1
