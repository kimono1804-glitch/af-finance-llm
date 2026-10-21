#!/usr/bin/env bash
# Stop hook: before Claude finishes a turn, make sure documentation/Changelog.md and
# documentation/memory.md were updated after the latest change to any project file.
# If not, block the stop and tell Claude which files changed, so it logs them first.
input="$(cat)"
# Avoid loops: if Claude is already continuing because of this hook, let it stop.
if printf '%s' "$input" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
  exit 0
fi

SELF="$0"; command -v cygpath >/dev/null 2>&1 && SELF="$(cygpath -u "$0")"
REPO="$(cd "$(dirname "$SELF")/../.." && pwd)" || exit 0
cd "$REPO" || exit 0
CL="documentation/Changelog.md"
MEM="documentation/memory.md"
[ -f "$CL" ] && [ -f "$MEM" ] || exit 0

# The older of the two docs is the reference point: both must be newer than every change.
REF="$CL"; [ "$MEM" -ot "$CL" ] && REF="$MEM"

changed="$(find . \
  \( -path ./.git -o -path ./documentation -o -path ./outputs -o -path ./unsloth_compiled_cache \
     -o -path ./vector_store -o -path ./chroma_db -o -path ./qdrant_storage -o -path ./.pytest_cache \
     -o -name __pycache__ \) -prune -o -type f -newer "$REF" -print 2>/dev/null \
  | grep -v -E '\.pyc$' | sed 's#^\./##' | head -n 25 | tr '\n' ' ')"

if [ -n "$changed" ]; then
  reason="Project files changed after the documentation was last updated: ${changed}. Before finishing, add timestamped entries for these changes at the top of documentation/Changelog.md and update documentation/memory.md (facts, file purposes, decisions, build progress %)."
  reason="$(printf '%s' "$reason" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  printf '{"decision":"block","reason":"%s"}\n' "$reason"
fi
exit 0
