from transformers import AutoTokenizer, AutoProcessor

def chat_template_v2(tokenizer) -> str:
    return tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},
                                          {'role':'previous text','content':'{previous}'},
                                          {'role':'user','content':'{message}'}],
                                                add_generation_prompt=True,
                                                tokenize=False)

# Modified Template for Video Context
TEMPLATE = """{TimeStamp} {VideoClip}"""

TEMPLATE_FINAL_BOXED = """{TimeStamp} {VideoClip}
Time={EndTime}s You are presented with a problem and a previous memory derived from watching a video. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

{PromptFinal}

Your answer:
"""

MEMORY_PROMPT = """[System]
You are a Streaming Video Analyst.

**Streaming Thinking Rules:**
1. **Update Only**: Read <previous_thought>. Only record **new** clues from the current segment relevant to <problem>. Do not repeat history.
2. **Wait for End**: Do not provide the final answer until the video stream is complete. Currently, just accumulate evidence

<problem>
{prompt}
</problem>

<previous_thought>
{memory}
</previous_thought>"""


cus_chat_template = chat_template_v2(processor)
token_message_template = cus_chat_template.format(message=TEMPLATE,previous=MEMORY_PROMPT)
print(token_message_template) # 非最后一轮，不要求回答问题

final_chat_template = chat_template_v2(processor)
token_message_template = final_chat_template.format(message=TEMPLATE_FINAL_BOXED,previous=MEMORY_PROMPT)
print(token_message_template) # 最后一轮，回答问题

