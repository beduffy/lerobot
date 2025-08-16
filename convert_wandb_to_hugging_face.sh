

# 1) Create the HF-format folder for 090000
mkdir -p /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/checkpoints/090000/pretrained_model

# 2) Download the 090000 weights from W&B
wandb artifact get --type model benfduffy-bearcover-gmbh/lerobot/policy_act-dataset_bearlover365_pick_place_one_white_sock_black_out_blinds-seed_1000-090000:v0 --root /teamspace/studios/this_studio/lerobot/tmp_wandb_090000

# 3) Move weights into the HF folder
cp /teamspace/studios/this_studio/lerobot/tmp_wandb_090000/model.safetensors \
   /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/checkpoints/090000/pretrained_model/model.safetensors

# 4) Add configs from your run (step-invariant)
cp /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/config.json \
   /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/checkpoints/090000/pretrained_model/config.json

cp /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/train_config.json \
   /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/checkpoints/090000/pretrained_model/train_config.json

# # 5) Upload in the same format you normally do
# huggingface-cli upload ${HF_USER}/${task_name} \
#   /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/checkpoints/090000/pretrained_model \
#   --repo-type=model --commit-message "090000 checkpoint reconstructed from W&B"


# Check vars (they must NOT be empty)
echo "HF_USER=${HF_USER}"; echo "task_name=${task_name}"

# If empty, set them
export HF_USER=your-hf-username
export task_name=pick_place_one_white_sock_black_out_blinds

# (Optional) verify login and repo exists
huggingface-cli whoami
huggingface-cli repo create ${HF_USER}/${task_name} --type model --yes || true

# Upload the reconstructed 090000 checkpoint
huggingface-cli upload ${HF_USER}/${task_name} \
  /teamspace/studios/this_studio/lerobot/outputs/train/pick_place_one_white_sock_black_out_blinds_1/checkpoints/090000/pretrained_model \
  --repo-type=model --commit-message "090000 checkpoint reconstructed from W&B"