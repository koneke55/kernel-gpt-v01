#!/usr/bin/env bash
"""
auto_push.sh - Create sequential commits and push them to a branch.

Usage:
  bash scripts/auto_push.sh --count 100 --branch my-auto-branch

Options:
  --count N           Number of commits/pushes to perform (default: 100)
  --branch NAME       Branch to push to (default: auto-push-<timestamp>)
  --remote NAME       Git remote name (default: origin)
  --delay SECONDS     Delay between pushes in seconds (default: 0.2)
  --author-name NAME  Optional commit author name to override local config
  --author-email EM   Optional commit author email to override local config
  --dry-run           Do everything except actually push to remote
  --help              Show this help

Notes:
  - This script does NOT store credentials. Use your normal git auth (ssh/credential helper).
  - Use with caution: avoid spamming public repos. Prefer a dedicated branch.
  - Appends to a local file named .autopush to guarantee a change each iteration.

"""
set -euo pipefail

COUNT=100
BRANCH=""
REMOTE=origin
DELAY=0.2
AUTHOR_NAME=""
AUTHOR_EMAIL=""
DRY_RUN=0
MESSAGE=""
NO_PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count)
      COUNT="$2"; shift 2;;
    --branch)
      BRANCH="$2"; shift 2;;
    --remote)
      REMOTE="$2"; shift 2;;
    --delay)
      DELAY="$2"; shift 2;;
    --author-name)
      AUTHOR_NAME="$2"; shift 2;;
    --author-email)
      AUTHOR_EMAIL="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift 1;;
    --message)
      MESSAGE="$2"; shift 2;;
    --no-push)
      NO_PUSH=1; shift 1;;
    --help)
      sed -n '1,180p' "$0"; exit 0;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$BRANCH" ]]; then
  BRANCH="auto-push-$(date +%Y%m%dT%H%M%S)"
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository. cd to the repo root and retry." >&2
  exit 2
fi

# If author not provided, try environment vars then git config
if [[ -z "$AUTHOR_NAME" ]]; then
  AUTHOR_NAME=${AUTHOR_NAME:-$(git config user.name || true)}
  AUTHOR_NAME=${AUTHOR_NAME:-${AUTHOR_NAME_ENV:-}}
fi
if [[ -z "$AUTHOR_EMAIL" ]]; then
  AUTHOR_EMAIL=${AUTHOR_EMAIL:-$(git config user.email || true)}
  AUTHOR_EMAIL=${AUTHOR_EMAIL:-${AUTHOR_EMAIL_ENV:-}}
fi

echo "Using author: ${AUTHOR_NAME:-(unset)} <${AUTHOR_EMAIL:-(unset)}>"

echo "Using branch: $BRANCH, remote: $REMOTE, count: $COUNT, delay: $DELAY, dry-run: $DRY_RUN, no-push: $NO_PUSH"

# create or switch to branch
if git show-ref --verify --quiet refs/heads/"$BRANCH"; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH"
fi

FILE=.autopush
touch "$FILE"

for i in $(seq 1 "$COUNT"); do
  echo "Auto push entry $i at $(date --iso-8601=seconds)" >> "$FILE"
  git add "$FILE"
  if [[ -n "$MESSAGE" ]]; then
    MSG="$MESSAGE (#$i)"
  else
    MSG="auto: push #$i"
  fi
  # Always include author if we have at least one piece of information
  if [[ -n "$AUTHOR_NAME" || -n "$AUTHOR_EMAIL" ]]; then
    # Build author field conservatively
    ANAME=${AUTHOR_NAME:-""}
    AEMAIL=${AUTHOR_EMAIL:-""}
    if [[ -n "$ANAME" && -n "$AEMAIL" ]]; then
      git commit --author="$ANAME <$AEMAIL>" -m "$MSG" || true
    elif [[ -n "$ANAME" ]]; then
      git commit --author="$ANAME <>" -m "$MSG" || true
    elif [[ -n "$AEMAIL" ]]; then
      git commit --author="<$AEMAIL>" -m "$MSG" || true
    fi
  else
    git commit -m "$MSG" || true
  fi

  if [[ "$DRY_RUN" -eq 0 && "$NO_PUSH" -eq 0 ]]; then
    echo "Pushing ($i/$COUNT) to $REMOTE/$BRANCH..."
    git push "$REMOTE" "$BRANCH"
  else
    echo "Skipping push ($i/$COUNT) (dry-run=$DRY_RUN no-push=$NO_PUSH)"
  fi

  sleep "$DELAY"
done

echo "Completed $COUNT commits on branch $BRANCH (dry-run=$DRY_RUN)"
