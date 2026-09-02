#!/bin/bash
cd /home/anon/apps/boat
/home/anon/apps/boat/venv/bin/python run_optimization.py --config config.yaml 2>&1 | while IFS= read -r line; do
  echo "$(date +"%Y-%m-%d %H:%M:%S") $line"
done >> output/pipeline.log