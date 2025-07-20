# which python 
# export CUDA_VISIBLE_DEVICES=1
# python examples/convert_json_to_lerobot.py --raw-dir /workspace/dataset/surgical_learning --repo-id cpw/test --robot-type DVRK 
# python examples/convert_json_to_lerobot.py --raw-dir /workspace/dataset/phantom_retraction_first_direction --repo-id cpw/real_world_retraction_first_direction_future_js_predict --robot-type DVRK 
# examples


# Bechmarking the training of a policy using the ALOHA dataset with DINOv2 backbone
# Check if there is a performance difference between number of decoder layers (original 1 vs paper 7)
# export CUDA_VISIBLE_DEVICES=1
# python lerobot/scripts/train.py \
#     --policy.type=act \
#     --dataset.repo_id=lerobot/aloha_sim_insertion_human \
#     --env.type=aloha \
#     --output_dir=outputs/train/act_aloha_7decoder_layers \
#     --job_name=act_aloha_insertion_7decoder_layers \
#     --wandb.enable=true


# Real-world retraction dataset training with default settings (Resnet)
export CUDA_VISIBLE_DEVICES=1 
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=cpw/real_world_retraction_future_js_predict \
    --output_dir=outputs/train/act_phantom_retraction_future_js_predict_dinov2 \
    --job_name=act_phantom_retraction_future_js_predict_dinov2 \
    --policy.vision_backbone=facebook/dinov2-with-registers-base \
    --wandb.enable=true


# Real-world retraction dataset training with SwinV2 backbone
# python lerobot/scripts/train.py \
#     --policy.type=act \
#     --dataset.repo_id=cpw/real_world_retraction \
#     --output_dir=outputs/train/act_phantom_retraction_swinv2 \
#     --job_name=act_phantom_retraction_swinv2 \
#     --policy.vision_backbone=microsoft/swinv2-tiny-patch4-window8-256 \
#     --wandb.enable=true


# Real-world retraction dataset training with DINOv2 backbone
# python lerobot/scripts/train.py \
#     --policy.type=act \
#     --dataset.repo_id=cpw/real_world_retraction \
#     --output_dir=outputs/train/act_phantom_retraction_dinov2 \
#     --job_name=act_phantom_retraction_dinov2 \
#     --policy.vision_backbone=facebook/dinov2-with-registers-base \
#     --wandb.enable=true

# export CUDA_VISIBLE_DEVICES=1
# python examples/eval_act.py