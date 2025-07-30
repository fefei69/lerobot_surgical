# which python 
# export CUDA_VISIBLE_DEVICES=1
# python examples/convert_json_to_lerobot.py --raw-dir /workspace/dataset/surgical_learning --repo-id cpw/test --robot-type DVRK 
# python examples/convert_json_to_lerobot.py --raw-dir /workspace/dataset/phantom_retraction_first_direction --repo-id cpw/real_world_retraction_first_direction_future_js_predict --robot-type DVRK 
python examples/convert_json_to_lerobot.py \
    --raw-dir /workspace/dataset/disp_toy_data \
    --repo-id cpw/test_disp \
    --robot-type DVRK 
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
# export CUDA_VISIBLE_DEVICES=1 
# python lerobot/scripts/train.py \
#     --policy.type=act \
#     --dataset.repo_id=cpw/real_world_retraction_js_30hz_concat_eff_vel_dep \
#     --output_dir=outputs/train/test \
#     --job_name=test_temp \
#     --wandb.enable=true


# python lerobot/scripts/train.py \
#     --policy.type=act \
#     --dataset.repo_id=cpw/real_world_retraction_js_30hz_concat_eff_vel_dep \
#     --output_dir=outputs/train/act_phantom_retraction_rel_ee_30hz_eff_vel_depthimg \
#     --job_name=act_phantom_retraction_rel_ee_30hz_eff_vel_depthimg \
#     --wandb.enable=true

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


# export CUDA_VISIBLE_DEVICES=1 
# python lerobot/scripts/train.py \
#     --policy.type=diffusion \
#     --dataset.repo_id=cpw/real_world_retraction_predict_js_30hz_concat_eff_vel \
#     --output_dir=outputs/train/diffusion_phantom_retraction_predict_js_30hz_eff_vel \
#     --job_name=diffusion_phantom_retraction_predict_js_30hz_eff_vel \
#     --wandb.enable=true