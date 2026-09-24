CKPT=/mnt/images-ks3-hd/workspace/guanyiran/videostreamingthink/MemAgent-main/video_memory_agent/0226_3b/global_step_40
BASE=/mnt/images-ks3-hd/models/lmm/qwenvl/Qwen2.5-VL-3B-Instruct
# BASE=Qwen/Qwen2.5-14B-Instruct

TARGET=$CKPT/huggingface
python3 scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path $BASE \
    --local_dir $CKPT/actor \
    --target_dir $TARGET
cp $BASE/token*json $TARGET
cp $BASE/vocab.json $TARGET
cp $BASE/preprocessor_config.json $TARGET
cp $BASE/chat_template.json $TARGET