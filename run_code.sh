# which python 
# python examples/convert_json_to_lerobot.py --raw-dir /workspace/dataset/surgical_learning --repo-id cpw/test --robot-type DVRK 
# examples

python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --output_dir=outputs/train/act_aloha_insertion_original \
    --wandb.enable=true




