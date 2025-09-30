#!/bin/bash
if [ ! -d "temp" ]; then
    mkdir temp
fi
# 定义最大并行数
MAX_JOBS=4

# 定义所有要运行的命令（数组）
COMMANDS=(
    "python main_md.py --model stegmn --exp_name stegmn-aspirin --mol aspirin"
    "python main_md.py --model stegmn --exp_name stegmn-benzene --mol benzene"
    "python main_md.py --model stegmn --exp_name stegmn-ethanol --mol ethanol"
    "python main_md.py --model stegmn --exp_name stegmn-malonaldehyde --mol malonaldehyde"
    "python main_md.py --model stegmn --exp_name stegmn-naphthalene --mol naphthalene"
    "python main_md.py --model stegmn --exp_name stegmn-salicylic --mol salicylic "
    "python main_md.py --model stegmn --exp_name stegmn-toluene --mol toluene "
    "python main_md.py --model stegmn --exp_name stegmn-uracil --mol uracil "
)

# 可选：为每个命令加日志重定向（避免输出混杂）
for i in "${!COMMANDS[@]}"; do
    COMMANDS[$i]+=" > temp/log_${i}.txt 2>&1"
done

# 启动队列式并行执行
running_jobs=0
for cmd in "${COMMANDS[@]}"; do
    if (( running_jobs >= MAX_JOBS )); then
        # 等待任意一个任务完成
        wait -n
        ((running_jobs--))
    fi
    # 启动新任务
    echo "🚀 Starting: $cmd"
    eval "$cmd" &
    ((running_jobs++))
done

# 等待剩余所有任务完成
wait

echo "✅ All tasks completed."