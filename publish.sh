#!/bin/bash
# Publish the generated report (html_output/) to www.avmapper.com/plusv-dangers/.
#
# The S3 bucket is shared with the METAR site (published by
# ~/projects/metar_calendar/website/frontend/_scripts/publish-to-s3.py, which
# excludes plusv-dangers/ from its own sync). This script only ever writes
# under the plusv-dangers/ prefix: --delete removes stale files there (e.g.
# charts from an earlier run that no longer qualify) and nothing else in the
# bucket. Bucket versioning is off, so deletions are permanent.
#
# Usage: ./publish.sh [--dryrun]   (--dryrun shows the changes without making them)

set -euo pipefail

AWS_PROFILE=jelson-personal
BUCKET=avmapper.com
PREFIX=plusv-dangers
DISTRIBUTION_ID=E39PVZW9UJ3RUQ
SRC="$(dirname "$0")/html_output"
export AWS_PROFILE

DRYRUN=()
case "${1:-}" in
    --dryrun) DRYRUN=(--dryrun) ;;
    "") ;;
    *) echo "usage: $0 [--dryrun]" >&2; exit 2 ;;
esac

# Syncing an empty or missing directory with --delete would wipe the
# published report
if [ ! -f "$SRC/index.html" ]; then
    echo "$SRC/index.html not found; generate the report first" >&2
    exit 1
fi

# Keep the trailing slash on the destination so the sync (and its --delete)
# is confined to the plusv-dangers/ folder
aws s3 sync --delete "${DRYRUN[@]}" "$SRC/" "s3://$BUCKET/$PREFIX/"

if [ ${#DRYRUN[@]} -eq 0 ]; then
    aws cloudfront create-invalidation --distribution-id "$DISTRIBUTION_ID" \
        --paths "/$PREFIX/*" --query 'Invalidation.[Id,Status]' --output text
fi
