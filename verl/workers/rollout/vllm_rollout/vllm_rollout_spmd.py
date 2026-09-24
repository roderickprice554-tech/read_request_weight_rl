# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.utils.dataset.vision_utils import process_image, process_video

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

import time
import datetime 
def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id: Union[int, List[int]], prompt_token_ids: torch.Tensor) -> List[int]:
    """
    1. 移除 prompt 左侧的 padding (支持单个或多个 pad_id)。
    2. 将连续的 151656 (video_pad) 合并为一个。
    """
    # ==========================================
    # 第一步：去除左侧 Padding (Left Padding Removal)
    # ==========================================
    
    # 1. 统一处理 pad_token_id 为 Tensor
    if isinstance(pad_token_id, int):
        pad_ids_list = [pad_token_id]
    else:
        pad_ids_list = pad_token_id
        
    pad_ids_tensor = torch.tensor(pad_ids_list, device=prompt_token_ids.device)

    # 2. 找到第一个 "非 Pad" 的位置
    # torch.isin 检查每个元素是否是 Pad
    is_pad_mask = torch.isin(prompt_token_ids, pad_ids_tensor)
    non_pad_indices = torch.nonzero(~is_pad_mask, as_tuple=False)

    if len(non_pad_indices) == 0:
        # 如果全是 Pad，返回空列表
        return []

    # 3. 切片，只保留有效内容
    start_index = non_pad_indices[0][0]
    trimmed_ids = prompt_token_ids[start_index:]

    # ==========================================
    # 第二步：合并连续的 151656 (Deduplication)
    # ==========================================
    
    target_id = 151656
    
    # 1. 标记所有等于 target_id 的位置
    is_target = (trimmed_ids == target_id)
    
    # 2. 构造 "前一个位置也是 target_id" 的掩码
    # 逻辑：将 is_target 向右移一位，第一位补 False (因为第一个元素前面没有元素)
    prev_is_target = torch.cat([
        torch.tensor([False], device=trimmed_ids.device), 
        is_target[:-1]
    ])
    
    # 3. 找出需要删除的元素：当前是 target 且 前一个也是 target
    should_remove = is_target & prev_is_target
    
    # 4. 保留不需要删除的元素 (~should_remove)
    final_ids = trimmed_ids[~should_remove]

    return final_ids.tolist()

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, List[Any]], repeats: int) -> Union[torch.Tensor, np.ndarray, List[Any]]:
    """
    Safely repeat data for Tensor, Numpy Array, and Python List (containing None or Objects).
    """
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    elif isinstance(value, list):
        return [item for item in value for _ in range(repeats)]
    else:
        raise TypeError(f"Unsupported type for _repeat_interleave: {type(value)}")



# 在 _repeat_interleave 函数后添加
def _process_multi_modal_data(
    multi_modal_data: dict, min_pixels: int, max_pixels: int, video_fps: float = 1.0
) -> dict:
    """处理多模态数据，支持 min_pixels 和 max_pixels 控制"""
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None

class vLLMRollout(BaseRollout):
    def __init__(self, 
        model_path: str, 
        config: DictConfig, 
        tokenizer, 
        model_hf_config,
        processor=None,
        model_vision_encoder=None,
        if_embed_cache=False,
        **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        self.if_embed_cache = False
        if if_embed_cache:
            self.if_embed_cache = True
            self.processor = processor
            self.model_vision_encoder = model_vision_encoder

        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192 * 2)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=False,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"default kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        ######
        # ADD
        ######
        if torch.distributed.get_rank() == 0:
            print(f"updating sampling params: {kwargs}")
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    #####
    # MODIFY: We want to specify a different value from sampling_params.max_tokens,
    #         since intermidiate turns and final turn may have different max_tokens.
    #         `pad_to` is used when it is passed, else we use sampling_params.max_tokens
    #####
    def generate_sequences(self, prompts: DataProto, pad_to=None, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()
        start_time = time.time()
        if torch.distributed.get_rank() == 0:
            print(f"{_now()} start")

        
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        uid = prompts.non_tensor_batch["uid"]
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        max_video_token_num = prompts.meta_info.get('max_video_token_num', 0)

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        batch_multi_modal_embeds = []
        if batch_multi_modal_data is not None:
            tmp_embed_cache = {}
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data , _uid in zip(non_tensor_batch.pop("raw_prompt_ids"), batch_multi_modal_data, uid):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
                if self.if_embed_cache and not is_validate:
                    if multi_modal_data is not None:
                        _device = self.model_vision_encoder.device
                        _dtype = multi_modal_data["video"][0].dtype
                        if _uid in tmp_embed_cache:
                            batch_multi_modal_embeds.append({"video": tmp_embed_cache[_uid]})
                        else:
                            raw_outputs = self.processor(text=["<|video_pad|>"], videos=multi_modal_data["video"][0], return_tensors="pt")
                            pixel_values_video = raw_outputs["pixel_values_videos"].type(self.model_vision_encoder.dtype).to(_device)     
                            grid_thw = raw_outputs["video_grid_thw"].to(_device)
                            with torch.no_grad():
                                visual_embed = self.model_vision_encoder(pixel_values_video, grid_thw=grid_thw).to(torch.bfloat16).cpu()
                            tmp_embed_cache[_uid] = visual_embed
                            batch_multi_modal_embeds.append({"video": visual_embed})
                    else:
                        batch_multi_modal_embeds.append({"video": None})

            del tmp_embed_cache 
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        #######
        # MODIFY: careful! we should update kwargs here, else the given **kwargs will be ignored
        #######
        if not do_sample:
            if torch.distributed.get_rank() == 0:
                print(f"original {kwargs=}, updating becase do_sample is False")
            kwargs.update({
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            })
        elif is_validate:
            if torch.distributed.get_rank() == 0:
                print(f"{_now()}original {kwargs=}, updating because is_validate is True")
            # TODO: try **
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            })

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            prepared_time = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"prepare time: {prepared_time - start_time}")
            
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            generated_time = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"{_now()}generate time: {time.time() - prepared_time}")

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            #####
            # MODIFY: we should pad by sampling_params instead of by config here.
            #####
            pad_to = pad_to if pad_to is not None else self.sampling_params.max_tokens
            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=pad_to).to(idx.device)
            if torch.distributed.get_rank() == 0:
                print(f"{_now()}pad time: {time.time() - generated_time}")

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                uid = _repeat_interleave(uid, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)
                
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        result_time = time.time()
        if torch.distributed.get_rank() == 0:
            print(f"result time: {time.time() - result_time}")
            print(f"total time: {time.time() - start_time}")

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if is_validate:
            non_tensor_batch = {"uid": uid}
        elif len(batch_multi_modal_embeds) > 0:
            batch_multi_modal_embeds = np.array(batch_multi_modal_embeds, dtype=object)
            non_tensor_batch = {
                "uid": uid,
                "multi_modal_embeds": batch_multi_modal_embeds,
            }
        elif batch_multi_modal_data is not None:
            non_tensor_batch = {
                "uid": uid,
                "multi_modal_data": batch_multi_modal_data,
            }
        else:
            non_tensor_batch = {}

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
