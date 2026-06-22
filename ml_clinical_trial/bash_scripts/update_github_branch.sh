#!/bin/bash

# Usage: ./git_push.sh "Commit message"
# If no commit message is provided, it uses the current date/time.

set -euo pipefail

# --- Resolve locations ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Your repository is one directory up from this script:
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Sanity check: ensure REPO_DIR is a git repo ---
if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: $REPO_DIR is not a Git repository."
  echo "Tip: make sure your script lives in a subfolder of the repo, or adjust REPO_DIR."
  exit 1
fi

# --- Configure Git user info (repo-scoped, not global) ---
git -C "$REPO_DIR" config user.name "Junfu Cheng"
git -C "$REPO_DIR" config user.email "junfu.cheng@ufl.edu"

# --- Branch name ---
BRANCH="main"

# --- Commit message ---
if [ $# -eq 0 ] || [ -z "${1:-}" ]; then
  COMMIT_MSG="Update on $(date '+%Y-%m-%d %H:%M:%S')"
else
  COMMIT_MSG="$1"
fi

# --- Switch to the branch (create if missing) ---
# Use 'switch' if available; fall back to 'checkout'
if git -C "$REPO_DIR" switch "$BRANCH" 2>/dev/null; then
  :
else
  # If branch doesn't exist, try creating it from current HEAD
  if git -C "$REPO_DIR" rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git -C "$REPO_DIR" checkout "$BRANCH"
  else
    git -C "$REPO_DIR" checkout -b "$BRANCH"
  fi
fi

# --- Stage & commit (skip if nothing changed) ---
git -C "$REPO_DIR" add -A

# Check for any staged or unstaged changes
if git -C "$REPO_DIR" diff --quiet && git -C "$REPO_DIR" diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git -C "$REPO_DIR" commit -m "$COMMIT_MSG"

# --- Push ---
git -C "$REPO_DIR" push -u origin "$BRANCH"
echo "Pushed to origin/$BRANCH from $REPO_DIR"