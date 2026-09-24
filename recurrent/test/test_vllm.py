import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# ==========================================
# 把主要逻辑封装到一个函数中，或者直接放在 main 块里
# ==========================================
def run_inference():
    model_path = "/mnt/images-ks3-hd/models/lmm/qwenvl/Qwen2.5-VL-3B-Instruct"
    video_path = "/root/videostreamingthink/vm_video_clip/_-W1fIDXBvs_4fps_480p/part_0.mp4"

    # 1. 初始化 Processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 2. 初始化 vLLM
    # 注意：LLM 的初始化必须在 main 保护块内部进行
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 1},
        gpu_memory_utilization=0.9,
    )

    # 3. 准备数据
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Describe this video."}
            ]
        }
    ]

    # 4. 计算 ID
    image_inputs, video_inputs = process_vision_info(messages)
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    prepared_prompt_token_ids = inputs.input_ids[0].tolist()

    # 5. 推理
    sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
    outputs = llm.generate(
        prompts={'prompt_token_ids': prepared_prompt_token_ids,
             'multi_modal_data': {'video': video_inputs[0]},},
        sampling_params=sampling_params
    )

    for output in outputs:
        print(output.outputs[0].text)

# ==========================================
# 关键修复：添加入口保护
# ==========================================
if __name__ == "__main__":
    run_inference()
