export PYTHONPATH=${PYTHONPATH}:./main/InternVL-2

python launch/launch_rm_zero3.py -c configs/qwen_configs/train_robust_8B.yaml -d checkpoints/robust_8B -f main/InternVL-2/train_rm_robust.py --gpus 16
