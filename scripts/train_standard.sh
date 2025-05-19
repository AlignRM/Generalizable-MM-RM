export PYTHONPATH=${PYTHONPATH}:./main/InternVL-2

python launch/launch_rm_zero3.py -c configs/qwen_configs/train_standard_8B.yaml -d checkpoints/standard_8B -f main/InternVL-2/train_rm.py --gpus 16
