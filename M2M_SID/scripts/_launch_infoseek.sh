#!/bin/bash
# INFOSEEK launcher wrapper
M2M_SID_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU=1
export CONFIG="$M2M_SID_ROOT/configs/infoseek.yaml"
export TAG=INFOSEEK
cd "$M2M_SID_ROOT"
exec bash "$M2M_SID_ROOT/scripts/run.sh"
