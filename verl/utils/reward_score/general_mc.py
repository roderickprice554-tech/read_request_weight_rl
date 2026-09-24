import re

def extract_first_option(text: str) -> str:
    """
    严格提取文本中出现的【第一个】单独的大写字母选项 (A-E)。
    防止 Reward Hacking（如模型输出 "A B C D" 来作弊）。
    """
    if not text:
        return None
    
    # 正则解释：
    # (?:^|\s|[^\w]) : 前缀必须是：字符串开头 OR 空格 OR 非单词字符(如括号、引号)
    # ([A-E])        : 捕获组，匹配 A-E
    # (?=$|\s|[^\w]) : 后缀必须是：字符串结尾 OR 空格 OR 非单词字符(如点、括号)
    pattern = re.compile(r"(?:^|\s|[^\w])([A-E])(?=$|\s|[^\w])")
    
    match = pattern.search(text) # 使用 search 只匹配第一个
    
    if match:
        return match.group(1).upper()
    return None

def get_model_answer(predict_str: str) -> str:
    """
    获取答案逻辑：
    1. 优先看 \\boxed{...} 内部，取框内的第一个字母。
    2. 如果没有框，看全文，取全文的第一个字母。
    """
    # 1. 尝试提取 \boxed{...}
    box_pattern = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
    box_match = box_pattern.search(predict_str)
    
    if box_match:
        # 提取框内内容，并在框内寻找第一个选项
        content = box_match.group(1)
        return extract_first_option(content)
    else:
        # 2. 【兜底】全文寻找第一个选项
        return extract_first_option(predict_str)

def format_reward(predict_str: str) -> float:
    """
    格式奖励：是否存在 \\boxed{}
    """
    pattern = re.compile(r"\\boxed\{.*\}", re.DOTALL)
    return 1.0 if pattern.search(predict_str) else 0.0

def acc_reward(predict_str: str, ground_truth: list) -> float:
    """
    准确率奖励
    """
    # 1. 提取预测
    pred_option = get_model_answer(predict_str)
    
    # 2. 提取真值 (处理 ['A.'] 或 'A' 的情况)
    gt_raw = ground_truth[0] if isinstance(ground_truth, list) and ground_truth else str(ground_truth)
    gt_option = extract_first_option(gt_raw)

    # 3. 比对
    if pred_option and gt_option and pred_option == gt_option:
        return 1.0
    return 0.0

def compute_score(predict_str: str, ground_truth: list, is_val=False) -> float:
    return acc_reward(predict_str, ground_truth)
    # if is_val:
    #     return acc_reward(predict_str, ground_truth)
    # else:
    #     return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(predict_str)

