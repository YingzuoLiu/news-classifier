#!/bin/bash

# 设置路径
PROJECT_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
PYTHON_EXEC=$(which python)

# 运行监控脚本
cd $PROJECT_DIR
$PYTHON_EXEC scripts/monitoring/model_monitor.py >> logs/monitoring.log 2>&1

echo "监控任务已完成，时间: $(date)"