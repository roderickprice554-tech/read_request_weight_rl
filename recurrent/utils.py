# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import re
import torch
import numpy as np
from transformers import PreTrainedTokenizer
from verl import DataProto
from tensordict import TensorDict # this will initilize CUDA! make sure your CUDA_VISIBLE_DEVICES is set!
from typing import List
import datetime
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class TokenTemplate:
    """
    format string, but in token_ids, use torch.LongTensor as data type.
    Input value can also be nunpy.ndarray or list[int].

    usage:
    ```
    TEMPLATE = "Here is a problem: {problem}"
    "Given this section: {section}"
    "Please answer it."

    processor = TokenTemplate(TEMPLATE, tokenizer)

    kwarg_text = dict(
        problem="What is the capital of France?",
        section="Here is a introduction to France. France is a country in Western Europe. Its capital is Paris.",
    )
    kwargs_token_ids = {
        k: tokenizer.encode(v, add_special_tokens=False) for k, v in kwarg_text.items()
    }

    print(tokenizer.decode(processor.format(**kwargs_token_ids)))

    # just as a text format string.
    assert TEMPLATE.format(**kwarg_text) == tokenizer.decode(processor.format(**kwargs_token_ids))
    ```
    """

    def __init__(self, template: str, tokenizer: PreTrainedTokenizer=None):
        self.template = template
        self.initialized = False
        if tokenizer:
            self.init(tokenizer)
        
    def init(self, tokenizer):
        self.keywords: list[str] = []  # Store extracted {keywords}
        self.token_sections: list[torch.LongTensor] = []  # Store tokenized text sections as LongTensors
        self.last_section: torch.LongTensor = None  # Last section as LongTensor
        
        # Match all {keywords}
        pattern = r'\{([a-zA-Z]+)\}'        
        parts = re.split(pattern, self.template)
        
        # Split text: even indices are non-{} parts, odd indices are {} keywords
        for i, part in enumerate(parts[:-1]):
            if i % 2 == 0:  # Even index, non-{} part
                tokens = tokenizer.encode(part, add_special_tokens=False)
                self.token_sections.append(torch.tensor(tokens, dtype=torch.long))
            else:  # Odd index, {} keyword
                self.keywords.append(part)
        self.last_section = torch.tensor(tokenizer.encode(parts[-1], add_special_tokens=False), dtype=torch.long)
        
        assert len(self.keywords) == len(self.token_sections), \
            f"{self.keywords} and {self.token_sections} should have the same length"
        self.initialized = type(tokenizer)

    @property
    def length(self) -> int:
        """
        Length of the template in token numbers
        """
        total = sum(section.numel() for section in self.token_sections)
        total += self.last_section.numel()
        return total

    def format(self, **kwargs: dict[str, torch.LongTensor | list[int] | np.ndarray]) -> torch.LongTensor:
        """
        Format the template with provided token ids
        
        Args:
            **kwargs: Dictionary of keyword to token ids (as LongTensor)
            
        Returns:
            Concatenated token ids as LongTensor
        """
        # Initialize with first section if exists
        formatted_parts = []
        
        # Reconstruct template by interleaving sections and keyword tokens
        for i, k in enumerate(self.keywords):
            if isinstance(kwargs[k], list):
                kwargs[k] = torch.tensor(kwargs[k], dtype=torch.long)
            elif isinstance(kwargs[k], np.ndarray):
                kwargs[k] = torch.from_numpy(kwargs[k].astype(np.int64)).to(torch.long)
            formatted_parts.append(self.token_sections[i])
            formatted_parts.append(kwargs[k])
        formatted_parts.append(self.last_section)
        
        return torch.cat(formatted_parts)

def chat_template(tokenizer, system=False) -> str:
    if system:
        return tokenizer.apply_chat_template([{'role':'system','content':'{system}'},
                                              {'role':'user','content':'{message}'}],
                                                    add_generation_prompt=True,
                                                    tokenize=False)
    else:
        return tokenizer.apply_chat_template([{'role':'user','content':'{message}'}],
                                                    add_generation_prompt=True, 
                                                    tokenize=False)

def chat_template_v2(tokenizer) -> str:
    return tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},
                                          {'role':'previous text','content':'{previous}'},
                                          {'role':'user','content':'{message}'}],
                                                add_generation_prompt=True,
                                                tokenize=False)
    

def graceful_padding(bsz: int, group_nums: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates an index mapping tensor that handles padding for grouped batches.
    
    The pattern is:
    - First group has no padding
    - Subsequent groups have 1 padding element
    - Padding elements are mapped to -1 (will be concatenated at the end)
    - Non-padding elements maintain their original order
    
    Example pattern for bsz=7, group_nums=3:
    no_padding_mask: [1, 1, 1, 0, 1, 1, 0, 1, 1]
    padding_index:   [0, 1, 2, -1, 3, 4, -1, 5, 6]
    
    Args:
        bsz: Batch size
        group_nums: Number of groups to split the batch into
        
    Returns:
        A tensor containing the index mapping with padding elements marked as -1
    """
    group_size = bsz // group_nums + 1
    remainder = bsz % group_nums
    if not remainder:
        return torch.arange(bsz), torch.ones(bsz, dtype=torch.bool)
    
    # Create mask where 1 = no padding, 0 = padding
    no_padding_mask = torch.tensor(
        [1 if i // group_size < remainder or i % group_size else 0 
         for i in range(group_nums * group_size)],
        dtype=torch.int
    )
    
    # Create cumulative index (shifted by -1)
    padding_index = torch.cumsum(no_padding_mask, dim=0) - 1
    
    # Mark padding elements with -1
    padding_index[~no_padding_mask.bool()] = -1
    
    return padding_index, no_padding_mask.bool()


# General torch utils. If input length exceeds max_length, this function does not truncate;
# it returns a tensor sized to the longest input sequence.
def pad_tensor_list_to_length(response: List[torch.LongTensor], pad_token_id, max_length=None, left_pad=True, return_mask=False):
    """
    similar to verl.utils.torch_functional.pad_2d_list_to_length 
    but 1. support left_pad 2. accept list[torch.Tensor] as input
    20x faster than pad_2d_list_to_length:
        - if use 2d list, simply create a tensor with shape(8192, 8192) will take 15s
        - if use list of 1d tensor, the whole process to pad(~8000->16384), concat, and stack will take only ~1s.
    """
    response_length = max(len(sub_list) for sub_list in response)
    if max_length is not None and max_length > response_length:
        target_length = max_length
    else:
        target_length = response_length
    full_long = lambda len, v: torch.full((len,), fill_value=v, dtype=response[0].dtype)
    if left_pad:
        padded_response = [torch.cat([full_long(target_length - len(sub_tensor), pad_token_id),
                                      sub_tensor]) for sub_tensor in response]     
    else:
        padded_response = [torch.cat([sub_tensor,
                                      full_long(target_length - len(sub_tensor), pad_token_id)
                                      ]) for sub_tensor in response]    
    padded_response = torch.stack(padded_response)
    if return_mask:
        mask = torch.full(padded_response.shape, True, dtype=torch.bool)
        if left_pad:
            [mask[i, :target_length - len(sub_tensor)].fill_(False) for i, sub_tensor in enumerate(response)]
        else:
            # [-0 : ] will be the whole tensor instead of empty tensor
            [mask[i, -(target_length - len(sub_tensor)):].fill_(False) 
             for i, sub_tensor in enumerate(response) if target_length - len(sub_tensor) > 0]
        return padded_response, mask
    else:
        return padded_response

def unpad(tokenizer, tensor: torch.Tensor, remove_eos: bool = False) -> np.ndarray:
    """Unpad tensor. Remove eos if specified"""
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0) 
    attention_mask = ~(tensor == tokenizer.pad_token_id)
    if remove_eos:
        attention_mask &= ~(tensor == tokenizer.eos_token_id)
    
    # Force object array format to avoid numpy's automatic conversion
    # when all tensors have the same length after unpadding
    result = np.empty(tensor.shape[0], dtype=object)
    for i in range(tensor.shape[0]):
        result[i] = tensor[i][attention_mask[i]]
    
    return result

def create_attention_mask(input_ids: torch.Tensor, pad_token_id) -> torch.Tensor:
    """Create attention mask from input ids."""
    return (input_ids != pad_token_id).to(torch.long)

def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Create position ids from attention mask."""
    return torch.clamp_min(torch.cumsum(attention_mask, dim=1) - 1, min=0)

import torch

def create_position_ids_vl(attention_mask: torch.Tensor, processor, vid_inputs, input_ids) -> torch.Tensor:
    """
    Build Qwen2-VL 3D position_ids.
    
    Args:
        attention_mask: (Batch, Seq_Len) - 0 for pad, 1 for valid tokens.
        input_ids: (Batch, Seq_Len) - already left-padded input ids.
        ...
    Returns:
        position_ids: (Batch, 3, Seq_Len) - Time, Height, Width channels.
    """
    from verl.models.transformers.qwen2_vl import get_rope_index
    
    pos_list = []
    
    for idx, item in enumerate(input_ids):
        # 1) Get per-sample mask (Seq_Len,).
        cur_mask = attention_mask[idx]
        
        # 2) Build 3D RoPE (3, Seq_Len).
        # get_rope_index typically handles video/image-specific logic.
        vid_pos_id = get_rope_index(
            processor=processor,
            input_ids=item,
            video_grid_thw=vid_inputs[idx].get("video_grid_thw", None), # Use .get for safer optional access.
            second_per_grid_ts=vid_inputs[idx].get("second_per_grid_ts", None),
            attention_mask=cur_mask, 
        )
        
        # 3) Safety step: force padding positions to zero.
        # With left-padding, leading pad tokens might otherwise get text-like positions.
        # Attention masking hides them during compute, but zeroing keeps positions clean.
        # cur_mask shape: (Seq_Len,) -> (1, Seq_Len) for broadcasting to 3 channels.
        vid_pos_id = vid_pos_id * cur_mask.unsqueeze(0)
        
        pos_list.append(vid_pos_id)

    # 4) Stack to (Batch, 3, Seq_Len).
    position_ids = torch.stack(pos_list, dim=0)
    
    return position_ids

    
def indexing_proto(proto: DataProto, indices: torch.Tensor | list | np.ndarray) -> DataProto:
    # make sure your fancy indices is supported by both np.ndarray and torch.Tensor
    # if indices is tensor, we should check device
    if isinstance(indices, torch.Tensor):
        cpu_indices = indices.cpu()
        if indices.device!= proto.batch.device:
            indices = indices.to(proto.batch.device)
    else:
        cpu_indices = indices
    return DataProto.from_dict(tensors={k: v[indices] for k, v in proto.batch.items()},
                               non_tensors={k: v[cpu_indices] for k, v in proto.non_tensor_batch.items()},
                               meta_info=proto.meta_info)

def slice_non_tensor_batch(non_tensor_batch: dict, indices: list[list[int]]) -> list[dict]:
    """
    Split non_tensor_batch by grouped indices.
    
    Args:
        non_tensor_batch: {"multi_modal_inputs": [...], ...}
        indices: [[0, 1], [2, 3, 4], ...]  # Sample indices for each micro-batch.
    
    Returns:
        list of non_tensor_batch dicts
    """
    return [
        {k: [v[i] for i in idx_group] for k, v in non_tensor_batch.items()}
        for idx_group in indices
    ]

def td_split(td: TensorDict, sections: int) -> list[TensorDict]:
    """
    split TensorDict in dim0, allows different sections, like torch.tensor_split and np.array_split
    used in workers/dp_actor to support variable length of batch size
    """
    if len(td) < sections:
        print(f"error occurred when trying to split {td}")
        raise ValueError(f"len(proto)={len(td)} < sections={sections}")        
    
    tensors_splitted = {k: torch.tensor_split(v, sections) for k, v in td.items()}
    return [TensorDict.from_dict({k: v[i] for k, v in tensors_splitted.items()}) for i in range(sections)]

def proto_split(proto: DataProto, sections: int) -> list[DataProto]:
    """Split DataProto into sections, handling both tensor and non_tensor batches."""
    # Split tensor batch
    batch_splits = td_split(proto.batch, sections)
    
    # Split non_tensor batch
    total_len = proto.batch.batch_size[0]
    indices_splits = np.array_split(np.arange(total_len), sections)
    
    non_tensor_splits = [
        {k: [v[i] for i in idx_chunk] for k, v in proto.non_tensor_batch.items()}
        for idx_chunk in indices_splits
    ]
    
    return [
        DataProto.from_dict(
            tensors=dict(batch_splits[i].items()),
            non_tensors=non_tensor_splits[i],
            meta_info=proto.meta_info
        )
        for i in range(sections)
    ]


def reverse_indices(tensor):
    """
    Return the unique elements of a tensor and their indices.
    https://discuss.pytorch.org/t/reverse-inverse-indices-torch-unique/114521/6
    return the reverse indices of an array
    ```py
    t[reverse_indices(t)] = unique(t)
    # e.g. [1, 4, 3, 2, 0] -> [4, 0, 3, 2, 1], 
    torch.tensor([1, 4, 3, 2, 0])[
          torch.tensor([4, 0, 3, 2, 1])
    ] == [0, 1, 2, 3, 4]
    ```

    Used in final_batch
    """
    unique, inverse_indices = torch.unique(tensor, return_inverse=True)
    assert len(unique) == len(tensor), f"Your input tensor has duplicated elements."
    indices = torch.scatter_reduce(
        torch.zeros_like(unique, dtype=torch.long, device=tensor.device), 
        dim=0,
        index=inverse_indices,
        src=torch.arange(tensor.size(0), device=tensor.device),
        reduce="amin",
        include_self=False,
    )
    return indices

def final_batch(batch: DataProto, final_mask: torch.Tensor, sample_index: torch.Tensor):
    """
    Extract the conversation that contains the final answer from the batch.
    Used in reward computation.
    """
    # 1. indexing by final_mask, find the final output
    # 2. indexing by reverse_index of final_index, reorder the output so that it matches the input order
    final_output = indexing_proto(batch, final_mask)
    final_index = sample_index[final_mask]
    final_output.reorder(reverse_indices(final_index))
    return final_output

def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[:target_len//2] + '\n\n...(truncated)\n\n' + string[-target_len//2:]

def log_step(logger, step, conversation):
    logger.info("="*30 + f"STEP {step}" + "="*30)
    for i, msg in enumerate(conversation):
        logger.info(f"[{msg['role']}]:")
        logger.info(f"{clip_long_string(msg['content'])}")
        logger.info("-"*50)

from openai.types.chat.chat_completion import Choice
def msg(choice: Choice):
    if isinstance(choice.stop_reason, str):
        stop_suffix = choice.stop_reason
    else:
        # Here is some possible stop_reason:
        # 1. None if eos_token is generated
        # 2. 151643 if pad_token is generated
        stop_suffix = ""
    return {
        "role": choice.message.role,
        "content": choice.message.content + stop_suffix,
        "finished": choice.finish_reason == "stop"
    }

from tqdm import tqdm  # Progress bar utility.

def chunked_inference(func, batch: DataProto, chunk_size: int, desc: str = "Inference", use_tqdm: bool = False):
    """
    Generic chunked inference with optional progress bar.
    
    Args:
        func: Remote callable (for example, actor.compute_log_prob).
        batch: Input DataProto.
        chunk_size: Chunk size.
        desc: Progress bar description text.
        use_tqdm: Whether to enable tqdm progress bar.
    """
    total_size = batch.batch.batch_size[0]
    
    # Run directly for small batches to avoid unnecessary progress overhead.
    if total_size <= chunk_size:
        return func(batch)

    # 1) Split inputs.
    num_chunks = (total_size + chunk_size - 1) // chunk_size
    input_chunks = batch.chunk(chunks=num_chunks)
    output_chunks = []

    # 2) Prepare iterator.
    iterator = input_chunks
    if use_tqdm:
        # unit='chunk' means progress is tracked per chunk.
        iterator = tqdm(input_chunks, desc=desc, total=num_chunks, unit="chunk")

    # 3) Execute loop.
    for i, mini_batch in enumerate(iterator):
        # Set boundary flags for first and final iteration.
        mini_batch.meta_info["is_first_iter"] = (i == 0)
        mini_batch.meta_info["is_final_iter"] = (i == num_chunks - 1)
        res = func(mini_batch)
        output_chunks.append(res)
        
    # 4) Merge outputs.
    if hasattr(DataProto, 'concat'):
        return DataProto.concat(output_chunks)
    else:
        # Fallback merge path.
        first_res = output_chunks[0]
        merged_dict = {}
        keys = first_res.batch.keys()
        for k in keys:
            merged_dict[k] = torch.cat([c.batch[k] for c in output_chunks], dim=0)
        return DataProto.from_dict(merged_dict, meta_info=first_res.meta_info)

import torch
from tqdm import tqdm

def chunked_inference_pipelined(func, batch: DataProto, chunk_size: int, 
                                 desc: str = "Inference", use_tqdm: bool = False):
    """
    Double-buffered pipelined inference: while computing chunk[i] on GPU,
    preload chunk[i+1] asynchronously.
    """
    total_size = batch.batch.batch_size[0]
    if total_size <= chunk_size:
        return func(batch)

    num_chunks = (total_size + chunk_size - 1) // chunk_size
    input_chunks = batch.chunk(chunks=num_chunks)
    output_chunks = []
    
    # Create a dedicated CUDA stream for async transfers.
    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()
    
    # Preload first chunk.
    current_chunk = input_chunks[0]
    current_chunk.meta_info["is_first_iter"] = True
    current_chunk.meta_info["is_final_iter"] = (num_chunks == 1)
    current_gpu = current_chunk.to(torch.cuda.current_device())
    
    iterator = range(num_chunks)
    if use_tqdm:
        iterator = tqdm(iterator, desc=desc, total=num_chunks, unit="chunk")
    
    next_gpu = None
    
    for i in iterator:
        # Asynchronously preload next chunk if present.
        if i + 1 < num_chunks:
            next_chunk = input_chunks[i + 1]
            next_chunk.meta_info["is_first_iter"] = False
            next_chunk.meta_info["is_final_iter"] = (i + 1 == num_chunks - 1)
            
            with torch.cuda.stream(transfer_stream):
                # Async copy to GPU (non_blocking=True).
                next_gpu = next_chunk.to(torch.cuda.current_device(), non_blocking=True)
            
            del next_chunk  # Release CPU-side reference.
        
        # Compute current chunk.
        res = func(current_gpu)
        output_chunks.append(res)
        
        del current_gpu  # Release current GPU tensor.
        
        # Wait for preload completion and switch buffer.
        if next_gpu is not None:
            transfer_stream.synchronize()  # Ensure transfer finished.
            current_gpu = next_gpu
            next_gpu = None
    
    # Merge outputs.
    if hasattr(DataProto, 'concat'):
        return DataProto.concat(output_chunks)
    else:
        first_res = output_chunks[0]
        merged_dict = {}
        for k in first_res.batch.keys():
            merged_dict[k] = torch.cat([c.batch[k] for c in output_chunks], dim=0)
        return DataProto.from_dict(merged_dict, meta_info=first_res.meta_info)

def get_cumulative_counts(x: torch.Tensor) -> torch.Tensor:

    perm = torch.argsort(x, stable=True)

    sorted_x = x[perm]

    unique_vals, counts = torch.unique_consecutive(sorted_x, return_counts=True)
    group_starts = torch.cat([torch.tensor([0], device=x.device), counts.cumsum(0)[:-1]])
    
    base_indices = torch.repeat_interleave(group_starts, counts)
    
    sorted_counts = torch.arange(len(x), device=x.device) - base_indices + 1
    
    result = torch.empty_like(sorted_counts)
    result[perm] = sorted_counts
    
    return result

def union_uid_clip_num(uid: np.ndarray, clip_num: torch.Tensor) -> torch.Tensor:
    out = []
    assert len(uid) == len(clip_num), "uid and clip_num must have the same length"
    for uid, clip_num in zip(uid, clip_num):
        out.append((uid,clip_num.item()))
    return np.array(out, dtype=object)