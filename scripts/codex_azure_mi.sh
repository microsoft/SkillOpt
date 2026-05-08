#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/azureuser/workspace-gzy/SkillReflection"
PY="/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python"
REAL_CODEX="/home/azureuser/.nvm/versions/node/v18.20.8/bin/codex"
CLIENT_ID="8cafa2b1-a2a7-4ad9-814a-ffe4aed7e800"
SCOPE="https://cognitiveservices.azure.com/.default"

token="$("$PY" - <<PY
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
cred = ManagedIdentityCredential(client_id="$CLIENT_ID")
print(get_bearer_token_provider(cred, "$SCOPE")())
PY
)"

export CODEX_HOME="${CODEX_HOME:-$ROOT/.codex_azure}"
export AZURE_OPENAI_AUTH_HEADER="Bearer $token"

exec "$REAL_CODEX" "$@"
