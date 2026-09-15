#!/usr/bin/env bash
# Wait for the E9 probe, then launch the 51-cell array only if it PASSED.
# The wait tolerates transient squeue failures: an empty reply is only believed after
# three consecutive misses, because a single blip previously ended a watcher early.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source ~/bottlecap/shared/scripts/ai/bottlecap_env.sh >/dev/null 2>&1
J="${1:?usage: e9_launch_on_pass.sh <probe-jobid>}"
miss=0
while :; do
  st=$(squeue -j "$J" -h -o "%T" 2>/dev/null)
  if [ -n "$st" ]; then miss=0; else miss=$((miss+1)); [ "$miss" -ge 3 ] && break; fi
  sleep 45
done
echo "=== PROBE $J FINISHED ==="
sacct -j "$J" --format=State,Elapsed,ExitCode -X -n
grep -E "registry OK|task_hash|version |score |delta|PROBE A" "logs/e9probe-$J.out" 2>/dev/null | tail -6
if grep -q "PROBE A PASS" "logs/e9probe-$J.out" 2>/dev/null; then
  ONE=$(awk '!/^#/ && NF {n++; if($4==1) printf "%s%d",(c++?",":""),n-1}' slurm/e9_cells.txt)
  TWO=$(awk '!/^#/ && NF {n++; if($4==2) printf "%s%d",(c++?",":""),n-1}' slurm/e9_cells.txt)
  echo "GATE PASSED -- launching 51 cells"
  sbatch --array="$ONE%12" --gres=gpu:1 slurm/e9_bench.sbatch
  sbatch --array="$TWO%3"  --gres=gpu:2 slurm/e9_bench.sbatch
else
  echo "GATE NOT PASSED -- array deliberately NOT launched"
  tail -c 1200 "logs/e9probe-$J.err" 2>/dev/null | tr '\r' '\n' | grep -viE "^$|it/s\]$" | tail -6
fi
