#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "Usage: ./infrastructure/create-pinecone-secret.sh pcsk-your-pinecone-api-key"
  exit 1
fi

REGION="${AWS_REGION:-us-west-1}"
SECRET_NAME="${PINECONE_SECRET_NAME:-awspineconeapikey1}"

if aws secretsmanager describe-secret --region "$REGION" --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value \
    --region "$REGION" \
    --secret-id "$SECRET_NAME" \
    --secret-string "$1"
else
  aws secretsmanager create-secret \
    --region "$REGION" \
    --name "$SECRET_NAME" \
    --secret-string "$1"
fi

