#!/usr/bin/env bash
# Hanzo Kata Coding Agent cURL
set -euo pipefail

API_HOST="${HANZO_API_HOST:-https://api.hanzo.ai}"
API_KEY="${HANZO_API_KEY:-}"

if [ -z "$API_KEY" ] && command -v hanzo >/dev/null 2>&1; then
  API_KEY="$(hanzo auth token 2>/dev/null || true)"
fi

AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
  AUTH_HEADER="Authorization: Bearer $API_KEY"
fi

curl -4 -s -X POST "$API_HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "$AUTH_HEADER" \
  -d '{
    "model": "zen",
    "messages": [
      {"role": "system", "content": "You are Hanzo Kata, an autonomous coding agent. Output minimal unified diffs."},
      {"role": "user", "content": "Fix python syntax error in main.py: print \"hello\""}
    ],
    "max_tokens": 512,
    "temperature": 0.0
  }' | jq -r '.choices[0].message.content // .choices[0].message.reasoning_content // .'
