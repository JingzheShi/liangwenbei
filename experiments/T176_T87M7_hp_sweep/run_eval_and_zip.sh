#!/bin/bash
# Run after training completes: eval all variants, build top-3 zips
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SCRIPT_DIR/eval_zip.log"; }

log "=== Running eval_variants.py ==="
cd "$SCRIPT_DIR"
python3 eval_variants.py 2>&1 | tee -a eval_zip.log

log "=== Running build_zips.py ==="
python3 build_zips.py 2>&1 | tee -a eval_zip.log

log "=== All done ==="
cat results.json
