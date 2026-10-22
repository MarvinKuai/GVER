#!/bin/bash
# OVEN launcher wrapper
M2M_SID_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU=0
export CONFIG="$M2M_SID_ROOT/configs/oven.yaml"
export TAG=OVEN
cd "$M2M_SID_ROOT"
exec bash "$M2M_SID_ROOT/scripts/run.sh"
