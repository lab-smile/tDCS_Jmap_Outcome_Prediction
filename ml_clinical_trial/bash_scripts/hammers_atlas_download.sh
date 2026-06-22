#!/usr/bin/env bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make a directory for the atlas
# Register for the user's information and download the atlas data in the following link:
# https://brain-development.org/brain-atlases/adult-brain-atlases/adult-brain-maximum-probability-map-hammers-mith-atlas-n30r83-in-mni-space/
# Create ../hammers_atlas relative to the script’s directory
mkdir -p "$SCRIPT_DIR/../hammers_atlas"

# Change into that directory
cd "$SCRIPT_DIR/../hammers_atlas" || exit 1
