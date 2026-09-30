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

import inspect
import logging
import os
import shutil
import tempfile

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def select_lora_state_dict(state_dict):
    return {name: tensor for name, tensor in state_dict.items() if "lora_" in name}


class FSDPVLLMShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        rollout=None,
        lora_rank: int = 0,
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.rollout = rollout
        self.lora_rank = int(lora_rank)
        self.lora_version = 0
        self.lora_root = None
        if self.lora_rank > 0:
            root = tempfile.mkdtemp(prefix="verl-vllm-lora-") if torch.distributed.get_rank() == 0 else None
            roots = [root]
            torch.distributed.broadcast_object_list(roots, src=0)
            self.lora_root = roots[0]

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        # Copy, not share memory
        load_format = "hf" if self.full_params else "dtensor"

        if self.lora_rank > 0:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()
            self.export_lora_adapter(params)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            torch.cuda.empty_cache()
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])
        elif vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
        else:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

            # update model params
            self.update_params(params)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            torch.cuda.empty_cache()

            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params):
        model = self.model_runner.model
        patch_vllm_moe_model_weight_loader(model)
        world_size = torch.distributed.get_world_size()
        loaded_params = model.load_weights(((name, param.full_tensor() if world_size != 1 and hasattr(param, "full_tensor") else param) for name, param in updated_params.items()))
        logger.info("vLLM load weights, loaded_params: %d", len(loaded_params))

    def export_lora_adapter(self, state_dict=None):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        if state_dict is None:
            state_dict = self.module.state_dict()
        sharded_adapter = select_lora_state_dict(state_dict)
        adapter_state = {
            name: tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
            for name, tensor in sharded_adapter.items()
        }
        version = self.lora_version
        adapter_path = os.path.join(self.lora_root, f"adapter-{version}")
        if torch.distributed.get_rank() == 0:
            peft_model = self.module.module
            os.makedirs(adapter_path, exist_ok=False)
            peft_model.peft_config["default"].save_pretrained(adapter_path)
            adapter_state = get_peft_model_state_dict(
                peft_model,
                state_dict=adapter_state,
                save_embedding_layers=False,
            )
            save_file(
                {name: tensor.detach().cpu().contiguous() for name, tensor in adapter_state.items()},
                os.path.join(adapter_path, "adapter_model.safetensors"),
            )
        torch.distributed.barrier()
        self.rollout.set_lora_adapter(adapter_path, version)
        logger.warning("Activated vLLM LoRA adapter version %d from %s", version, adapter_path)
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0 and version >= 2:
            shutil.rmtree(os.path.join(self.lora_root, f"adapter-{version - 2}"))
        self.lora_version += 1
