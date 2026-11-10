#!/usr/bin/env bash
# Fetch the source PDFs into data/ and verify them by checksum.
# The PDFs are third-party documents, so they are not committed to the repo.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

CC_FILE="data/cloudcompare_user_manual.pdf"
CC_URL="https://cloud.sdsc.edu/v1/AUTH_opentopography/www/shortcourses%2FA2HRT_RCN%2FCloudCompare%20v2.6.1%20-%20User%20manual.pdf"
CC_SHA="8d354b6abb641449107132c3addff802c567623a25b4b09aa0a21d2f795c03da"

LIDAR_FILE="data/lidar_processing_procedures.pdf"
LIDAR_SHA="01394deff50cf1fda393c9b88d9aa207c475b52bf2ff50c27f7290707d0f7fe7"

verify() {  # file, expected sha256
    local actual
    actual=$(sha256sum "$1" | cut -d' ' -f1)
    if [[ "$actual" == "$2" ]]; then
        echo "  OK  $1"
    else
        echo "  CHECKSUM MISMATCH  $1 (eval page numbers may not line up)" >&2
        return 1
    fi
}

if [[ ! -f "$CC_FILE" ]]; then
    echo "Downloading CloudCompare v2.6.1 user manual (10 MB)..."
    curl -fL --progress-bar -o "$CC_FILE" "$CC_URL"
fi
verify "$CC_FILE" "$CC_SHA"

if [[ -f "$LIDAR_FILE" ]]; then
    verify "$LIDAR_FILE" "$LIDAR_SHA"
else
    cat <<EOF
  MISSING  $LIDAR_FILE
    "Procedures for Processing LIDAR Point Cloud Files to Create Digital Elevation
    Models, Contours, and Elevation Changes" (UNH CCOM, v1.0, 23 pages).
    No stable public URL is known; place your copy at $LIDAR_FILE.
    The pipeline also runs without it (its eval questions will just miss).
EOF
fi
