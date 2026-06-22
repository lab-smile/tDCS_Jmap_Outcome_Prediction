#!/bin/bash

# Usage:
# ./edit_git_config_in_script.sh "Your Name" "your.email@example.com" "branch_name"

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 \"Your Name\" \"your.email@example.com\" \"branch_name\""
    exit 1
fi

NAME_REPL="$1"
EMAIL_REPL="$2"
BRANCH_REPL="$3"
TARGET_SCRIPT="update_github_branch.sh"

# -----------------------------
# Resolve this script's directory
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PATH="${SCRIPT_DIR}/${TARGET_SCRIPT}"

# Check if the target script exists
if [ ! -f "$TARGET_PATH" ]; then
    echo "[ERROR] Target script '$TARGET_PATH' not found."
    exit 1
fi

# Backup the original script
cp "$TARGET_PATH" "${TARGET_PATH}.bak"
echo "[INFO] Backup created at ${TARGET_PATH}.bak"

# Update git user.name, user.email and branch lines using sed
# name
sed -i'' -E \
  "s|^git[[:space:]]+-C[[:space:]]+([^[:space:]]+)[[:space:]]+config[[:space:]]+user\.name[[:space:]].*$|git -C \1 config user.name \"$NAME_REPL\"|" \
  "$TARGET_PATH"

# email
sed -i'' -E \
  "s|^git[[:space:]]+-C[[:space:]]+([^[:space:]]+)[[:space:]]+config[[:space:]]+user\.email[[:space:]].*$|git -C \1 config user.email \"$EMAIL_REPL\"|" \
  "$TARGET_PATH"

# branch (unchanged pattern)
sed -i'' -E \
  "s|^BRANCH=.*|BRANCH=\"$BRANCH_REPL\"|" \
  "$TARGET_PATH"

# Confirm the changes
echo "[INFO] Updated Git config and branch in $TARGET_PATH:"
grep -nE 'config user\.(name|email)|^BRANCH=' "$TARGET_PATH" || true
