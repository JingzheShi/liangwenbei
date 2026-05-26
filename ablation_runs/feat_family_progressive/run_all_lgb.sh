#!/bin/bash
# Run 25 LGB trainings: L2..L6 x 5 seeds
set -e
cd /root/projects/liangwenbei_workdir

ROOT=ablation_runs/feat_family_progressive
PROGRESS=worker-progress.json
LOG=$ROOT/run_all_lgb.log
RESULTS_CSV=$ROOT/lgb_results.csv

echo "level,seed,feat_dim,val_pnl,test_pnl,best_iter,train_time" > $RESULTS_CSV

T0=$(date +%s)
DONE=0
TOTAL=25

for LVL in L2 L3 L4 L5 L6; do
  for SEED in 1 2 3 4 5; do
    OUTDIR=$ROOT/${LVL}_lgb_s${SEED}
    if [ -f "$OUTDIR/results.json" ]; then
      echo "[skip] $LVL seed=$SEED (results.json exists)" >> $LOG
    else
      echo "[run] $LVL seed=$SEED at $(date -Is)" >> $LOG
      CUDA_VISIBLE_DEVICES=0 python3 $ROOT/train_progressive.py \
        --model lgb --level $LVL --seed $SEED \
        --out $OUTDIR --horizon 60 --gpu --no-wandb >> $LOG 2>&1 || \
        echo "[ERROR] $LVL seed=$SEED" >> $LOG
    fi
    if [ -f "$OUTDIR/results.json" ]; then
      python3 -c "
import json
d = json.load(open('$OUTDIR/results.json'))
print(','.join(str(d.get(k,'')) for k in ['level','seed','feat_dim','val_pnl_h60_sym','test_pnl_h60_sym','best_iter','train_time_sec']))
" >> $RESULTS_CSV
    fi
    DONE=$((DONE+1))
    if [ $((DONE % 5)) -eq 0 ] || [ $DONE -eq $TOTAL ]; then
      WT=$(( ($(date +%s) - T0) / 60 ))
      python3 -c "
import json
prog = {'status': 'running', 'step': 'LGB $LVL seed=$SEED done; $DONE/$TOTAL', 'progress': '$DONE/$TOTAL', 'wall_time_min': $WT, 'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)'}
json.dump(prog, open('$PROGRESS','w'), indent=2)
"
    fi
  done
done

WT=$(( ($(date +%s) - T0) / 60 ))
python3 -c "
import json
prog = {'status': 'lgb_trainings_done', 'step': 'all 25 LGB trainings done', 'progress': '25/25', 'wall_time_min': $WT, 'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)'}
json.dump(prog, open('$PROGRESS','w'), indent=2)
"
echo "[done] total $WT min" >> $LOG
