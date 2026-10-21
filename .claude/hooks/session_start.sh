#!/usr/bin/env bash
# SessionStart hook: load the project memory and recent changelog into Claude's context,
# so a new session starts from documentation/ instead of re-reading the whole codebase.
SELF="$0"; command -v cygpath >/dev/null 2>&1 && SELF="$(cygpath -u "$0")"
REPO="$(cd "$(dirname "$SELF")/../.." && pwd)" || exit 0
DOCS="$REPO/documentation"

echo "PROJECT MEMORY (auto-loaded by .claude/hooks/session_start.sh)."
echo "Use these files instead of analysing the codebase. Open source files only when a task needs their exact code."
echo "Repo: $REPO"
echo
if [ -f "$DOCS/memory.md" ]; then
  echo "===== documentation/memory.md ====="
  cat "$DOCS/memory.md"
else
  echo "(documentation/memory.md is missing - recreate it)"
fi
echo
if [ -f "$DOCS/Changelog.md" ]; then
  total=$(wc -l < "$DOCS/Changelog.md")
  echo "===== documentation/Changelog.md (newest 200 of $total lines; read the file for older entries) ====="
  # Newest entries are at the top of the file.
  head -n 200 "$DOCS/Changelog.md"
else
  echo "(documentation/Changelog.md is missing - recreate it)"
fi
exit 0
