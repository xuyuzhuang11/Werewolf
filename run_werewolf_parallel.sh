#!/bin/bash
mkdir -p ./logs
API_SERVER=1

echo $API_SERVER
date

for i in {1..5}
do
  timeout 7200s bash -c "
    echo \"Combat $i...\"
    python run_werewolf.py \
      --current-game-number $i \
      --message-window 15 \
      --answer-topk 5 \
      --retri-question-number 5 \
      --exps-retrieval-threshold 0.80 \
      --similar-exps-threshold 0.01 \
      --max-tokens 100 \
      --temperature 0.3 \
      --use-api-server $API_SERVER \
      --environment-config ./examples/werewolf.json \
      --role-config ./config/1.json \
      > ./logs/$i.con 2>&1
    echo \"Combat $i...OK!\"
  " &
done

wait
