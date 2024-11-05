#!/bin/bash

# 预先定义的name参数列表
NAME_LIST=("maychelle_en" "maychelle_cn")

# 循环处理每个name参数
for NAME in "${NAME_LIST[@]}"; do
  # 执行quick_start.sh并传入name参数
  ./quick_start.sh $NAME
done 