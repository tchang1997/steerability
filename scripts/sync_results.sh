#!/bin/bash

set -euo pipefail

# Check required env variables
: "${ACCOUNT_ID:?ACCOUNT_ID env var is required}"
: "${R2_BUCKET:?R2_BUCKET env var is required}"
: "${AWS_PROFILE:?AWS_PROFILE env var is required}"

# Optional: set default bucket and prefixes
# FUTURE: different prefixes by dataset
PREFIX_JSON="json/"
PREFIX_CSV="csv/"
ENDPOINT="https://${ACCOUNT_ID}.r2.cloudflarestorage.com"

# Dry run flag
DRYRUN=""
if [[ "${1:-}" == "--dryrun" ]]; then
  DRYRUN="--dryrun"
  echo "Dry-run enabled — no changes will be made."
fi

echo "Syncing JSON files to R2..."
cd results
aws s3 sync ./steerability_metrics "s3://${R2_BUCKET}/${PREFIX_JSON}" \
  $DRYRUN \
  --profile "$AWS_PROFILE" \
  --endpoint-url "$ENDPOINT" \
  --checksum-algorithm CRC32

echo "Syncing CSV files to R2..."
aws s3 sync ./judged "s3://${R2_BUCKET}/${PREFIX_CSV}" \
  $DRYRUN \
  --profile "$AWS_PROFILE" \
  --endpoint-url "$ENDPOINT" \
  --checksum-algorithm CRC32
cd ..

echo "Sync complete."

