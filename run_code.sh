# which python 
# python examples/convert_json_to_lerobot.py --raw-dir /workspace/dataset/surgical_learning --repo-id cpw/test --robot-type DVRK 
python examples/convert_json_to_lerobot.py --raw-dir /workspace/dataset/phantom_retraction --repo-id cpw/real_world_retraction --robot-type DVRK 
# examples

# python lerobot/scripts/train.py \
#     --policy.type=act \
#     --dataset.repo_id=lerobot/aloha_sim_insertion_human \
#     --env.type=aloha \
#     --output_dir=outputs/train/act_aloha_dinov2 \
#     --policy.vision_backbone=facebook/dinov2-with-registers-base \
#     --wandb.enable=true




