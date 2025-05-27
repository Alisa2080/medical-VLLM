# import json
# import os
# import time
# from zhipuai import ZhipuAI

# # --- 配置区域 ---
# ZHIPUAI_API_KEY = "ed19653c23954f32ab3d7819495b2b6a.MPxFBwFu9vjoi0c3"  # 在这里填入您的API Key
# MODEL_NAME = "glm-4-plus"  # 或 "glm-4-plus", "glm-3-turbo" 等，根据您的需求和可用性选择

# # 输入和输出目录
# # 这些应该是您第一个脚本处理后得到的JSON文件所在目录
# # 以及您希望存放二次清洗后JSON文件的目录
# INPUT_JSON_DIR = r'F:\dataset\medical_text_json'  # 第一个脚本的输出目录
# OUTPUT_CLEANED_JSON_DIR = r'F:\dataset\medical_text_json\version2' # GLM清洗后的输出目录

# # API调用之间的可选延迟（秒），以避免过于频繁的请求
# API_CALL_DELAY = 0.5  # 设置为0则不延迟

# # --- Prompt 设计 ---
# # 您可以根据需要调整这个系统Prompt
# SYSTEM_PROMPT = """你是一位专业的文本编辑和数据清洗专家，尤其擅长处理医学科研文献。
# 你的任务是仔细审查并优化用户提供的从科研论文中提取的文本片段。
# 请严格按照以下规则操作：
# 1.  主要目标是提高文本作为语言模型训练数据的质量。
# 2.  移除任何残留的、无意义的引用标记，例如空的方括号 `[]`，或仅包含逗号、分号的方括号 `[,]`, `[;]` 等。但要保留文本中合法的方括号用法（如有）。
# 3.  修正明显的语法错误和拼写错误，但必须保持原文的科学含义和专业术语不变。
# 4.  确保句子结构完整，文法通顺，逻辑连贯。
# 5.  移除任何非正式的注释、无关的元数据片段、或非文本内容（如图像占位符、未被完全清除的XML/HTML残留标签等）。
# 6.  【重要】不要删除或修改任何科学术语、实验数据、统计结果、基因/蛋白质名称等专业内容，除非它明显是一个录入错误且你能确定正确的形式。
# 7.  【重要】不要添加任何原始文本中没有的信息或你自己的解释、评论。
# 8.  如果文本片段本身非常短（例如少于5个词），或者明显是一个无意义的、无法修复的片段（例如随机字符、损坏的文本），请返回一个空字符串。
# 9.  如果文本片段看起来已经是高质量的、干净的科研文本，无需做大的改动，请直接返回原文。
# 10. 你的输出应该是且仅是清洗和优化后的文本字符串。不要包含任何额外的解释或对话。"""

# client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

# def clean_text_with_glm_api(text_segment: str) -> str:
#     """
#     使用GLM API清洗单个文本片段。
#     """
#     if not text_segment or not text_segment.strip():
#         return ""

#     try:
#         response = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": f"请清洗以下文本片段：\n\n\"{text_segment}\""}
#             ],
#             temperature=0.1, #较低的温度使输出更具确定性
#             max_tokens=2048  # 根据文本片段长度和期望输出来调整
#         )
        
#         # 检查是否有有效的 choices 和 message
#         if response.choices and response.choices[0].message:
#             cleaned_text = response.choices[0].message.content
#             return cleaned_text.strip() if cleaned_text else ""
#         else:
#             print(f"Warning: API response for segment did not contain expected structure. Segment: '{text_segment[:50]}...'")
#             return text_segment # 或者返回空字符串，表示清洗失败
            
#     except Exception as e:
#         print(f"Error calling GLM API for segment: '{text_segment[:50]}...': {e}")
#         # 在出错时，可以选择返回原始文本，或空字符串，或尝试重试
#         return text_segment # 返回原始文本，避免数据丢失

# def process_json_files_with_glm(input_dir: str, output_dir: str):
#     """
#     遍历输入目录中的JSON文件，使用GLM API清洗每个文本单元，
#     并将结果保存到输出目录。
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     json_files_to_process = [f for f in os.listdir(input_dir) if f.endswith(".json")]
#     total_files = len(json_files_to_process)
#     print(f"Found {total_files} JSON files in '{input_dir}' for GLM cleaning.")

#     for i, filename in enumerate(json_files_to_process):
#         input_file_path = os.path.join(input_dir, filename)
#         output_file_path = os.path.join(output_dir, filename) # 使用相同的文件名

#         # 避免重复处理已存在的文件（可选）
#         # if os.path.exists(output_file_path):
#         #     print(f"Skipping {filename}, output file already exists.")
#         #     continue

#         print(f"Processing file {i+1}/{total_files}: {filename}...")
        
#         try:
#             with open(input_file_path, 'r', encoding='utf-8') as f:
#                 original_text_units = json.load(f)
#         except Exception as e:
#             print(f"Error reading or parsing JSON file {input_file_path}: {e}")
#             continue

#         cleaned_text_units = []
#         for j, unit in enumerate(original_text_units):
#             print(f"  Cleaning segment {j+1}/{len(original_text_units)} for {filename}...")
#             if isinstance(unit, str) and unit.strip(): # 确保是需要清洗的字符串
#                 cleaned_unit = clean_text_with_glm_api(unit)
#                 if cleaned_unit: # 只有当清洗后文本非空时才添加
#                     cleaned_text_units.append(cleaned_unit)
#                 else:
#                     print(f"    Segment resulted in empty string after cleaning (original: '{unit[:50]}...')")

#                 if API_CALL_DELAY > 0:
#                     time.sleep(API_CALL_DELAY) # 等待，避免API过载
#             else: # 如果单元不是字符串或为空，直接跳过或按需处理
#                 print(f"    Skipping non-string or empty segment: {unit}")

#         if cleaned_text_units:
#             try:
#                 with open(output_file_path, 'w', encoding='utf-8') as f:
#                     json.dump(cleaned_text_units, f, ensure_ascii=False, indent=2)
#                 print(f"  Successfully saved cleaned text ({len(cleaned_text_units)} units) to {output_file_path}")
#             except Exception as e:
#                 print(f"Error writing cleaned JSON to {output_file_path}: {e}")
#         else:
#             print(f"  No valid text units remained after cleaning for {filename}.")

# if __name__ == '__main__':
#     if ZHIPUAI_API_KEY == "YOUR_OWN_ZHIPUAI_API_KEY" or not ZHIPUAI_API_KEY:
#         print("错误：请在脚本中设置您的 ZHIPUAI_API_KEY。")
#     elif not os.path.isdir(INPUT_JSON_DIR):
#         print(f"错误：找不到输入目录 '{INPUT_JSON_DIR}'。请确保路径正确，并且其中包含由第一阶段清洗产生的JSON文件。")
#     else:
#         print(f"--- Starting GLM Secondary Cleaning ---")
#         print(f"Input directory: {INPUT_JSON_DIR}")
#         print(f"Output directory: {OUTPUT_CLEANED_JSON_DIR}")
#         print(f"Using model: {MODEL_NAME}")
#         if API_CALL_DELAY > 0:
#             print(f"Delay between API calls: {API_CALL_DELAY} seconds")
        
#         process_json_files_with_glm(INPUT_JSON_DIR, OUTPUT_CLEANED_JSON_DIR)
#         print(f"--- GLM Secondary Cleaning Finished ---")

import json
import os
import time
from zhipuai import ZhipuAI
import re

# --- 配置区域 ---
ZHIPUAI_API_KEY = "ed19653c23954f32ab3d7819495b2b6a.MPxFBwFu9vjoi0c3"  # 在这里填入您的API Key
MODEL_NAME = "glm-4-plus"  # 或 "glm-4-plus", "glm-3-turbo" 等

INPUT_JSON_DIR = r'F:\dataset\medical_text_json\version1'
OUTPUT_CLEANED_JSON_DIR = r'F:\dataset\medical_text_json\version3' # 使用新的输出目录以区分

API_CALL_DELAY = 0.5

SYSTEM_PROMPT =   """你是一位专业的文本编辑和数据清洗专家，尤其擅长处理医学科研文献。
你的任务是仔细审查并优化用户提供的从科研论文中提取的文本片段。
请严格按照以下规则操作：
1.  主要目标是提高文本作为语言模型训练数据的质量。
2.  移除任何残留的、无意义的引用标记，例如空的方括号 `[]`，或仅包含逗号、分号的方括号 `[,]`, `[;]` 等。但要保留文本中合法的方括号用法（如有）。
3.  修正明显的语法错误和拼写错误，但必须保持原文的科学含义和专业术语不变。
4.  确保句子结构完整，文法通顺，逻辑连贯。
5.  移除任何非正式的注释、无关的元数据片段、或非文本内容（如图像占位符、未被完全清除的XML/HTML残留标签等）。
6.  【重要】不要删除或修改任何科学术语、实验数据、统计结果、基因/蛋白质名称等专业内容，除非它明显是一个录入错误且你能确定正确的形式。
7.  【重要】不要添加任何原始文本中没有的信息或你自己的解释、评论。
8.  如果文本片段本身非常短（例如少于5个词），或者明显是一个无意义的、无法修复的片段（例如随机字符、损坏的文本），请返回一个空字符串。
9.  如果文本片段看起来已经是高质量的、干净的科研文本，无需做大的改动，请直接返回原文。
10. 你的输出应该是且仅是清洗和优化后的文本字符串。不要包含任何额外的解释或对话。
11. 请勿将纯文本变量（如 'yc', 'pc'）转换为LaTeX或其他数学标记格式，除非输入文本中已明确存在此类格式。
12. 如果一个文本片段（或在长文本拆分后的子片段）其主要内容是程序代码、伪代码、算法步骤的直接罗列（而非对算法的文字性描述）、或者主要是数学变量/符号和公式的定义列表，而不是连贯的自然语言散文段落，那么这个片段应该被视作不适合训练的内容。
13. 如果输入的文本片段过长（例如，超过200个词），请尝试将其智能地拆分成多个更短的、语义连贯的子段落，每个子段落长度尽量在50到150词之间。
14. 你的最终输出必须是一个JSON对象，格式如下：
    {"cleaned_segments": ["分段1的内容（已清洗）", "分段2的内容（已清洗）", ...]}
    如果原始文本不需要拆分，则列表中只有一个元素。
    如果根据规则8判断文本应被丢弃，则返回：
    {"cleaned_segments": []}
    不要输出任何其他解释或对话，仅输出指定格式的JSON对象。"""

client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

def clean_text_with_glm_api(text_segment: str) -> str:
    """
    使用GLM API清洗单个文本片段，并移除返回内容中可能存在的外层引号。
    """
    if not text_segment or not text_segment.strip():
        return []

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请清洗以下文本片段：\n\n\"{text_segment}\""}
            ],
            temperature=0.1,
            max_tokens=3000,
            response_format={'type': 'json_object'}
        )
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            try:
                # API返回的content应该是一个JSON字符串，需要解析
                result_json_str = response.choices[0].message.content
                # 移除可能由模型错误添加的Markdown代码块标记
                if result_json_str.startswith("```json"):
                    result_json_str = result_json_str[len("```json"):]
                if result_json_str.endswith("```"):
                    result_json_str = result_json_str[:-len("```")]
                result_json_str = result_json_str.strip()

                data = json.loads(result_json_str)
                
                if isinstance(data, dict) and "cleaned_segments" in data and isinstance(data["cleaned_segments"], list):
                    # 进一步清理每个返回的片段中可能的外层引号
                    final_segments = []
                    for seg in data["cleaned_segments"]:
                        if isinstance(seg, str):
                            stripped_seg = seg.strip()
                            while len(stripped_seg) >= 2 and \
                                  ((stripped_seg.startswith('"') and stripped_seg.endswith('"')) or \
                                   (stripped_seg.startswith("'") and stripped_seg.endswith("'"))):
                                stripped_seg = stripped_seg[1:-1].strip()
                            if stripped_seg: #确保不是空字符串
                                final_segments.append(stripped_seg)
                    return final_segments
                else:
                    print(f"Warning: API response JSON did not match expected structure. Response: {result_json_str[:100]}...")
                    # 如果结构不对，但原始文本还行，可以考虑返回原始文本作为一个片段（需清洗）
                    # 或者返回空列表表示处理失败
                    # cleaned_original = clean_text_with_glm_api_basic(text_segment) # 调用一个只做清洗不做拆分的版本
                    # return [cleaned_original] if cleaned_original else []
                    return [] # 或者更简单地返回空

            except json.JSONDecodeError as je:
                print(f"Error decoding JSON response from API: {je}. Response: {response.choices[0].message.content[:200]}...")
                return [] # 返回空列表表示处理失败
            except Exception as ex:
                print(f"Error processing structured API response: {ex}. Response: {response.choices[0].message.content[:200]}...")
                return []
        else:
            print(f"Warning: API response for segment did not contain expected content. Segment: '{text_segment[:50]}...'")
            return [text_segment] if text_segment.strip() else [] # 返回原始文本（如果非空）作为一个片段
            
    except Exception as e:
        print(f"Error calling GLM API for segment: '{text_segment[:50]}...': {e}")
        # 返回原始文本作为一个片段，避免数据丢失，但它未被清洗或拆分
        return [text_segment] if text_segment.strip() else []

def process_json_files_with_glm(input_dir: str, output_dir: str):
    """
    遍历输入目录中的JSON文件，使用GLM API清洗每个文本单元，
    并将结果保存到输出目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_files_in_dir = os.listdir(input_dir)
    json_files_to_process_unsorted = [f for f in all_files_in_dir if f.endswith(".json")]
    
    # 2. 定义一个排序键函数，用于提取文件名中的数字部分
    def sort_key_natural(filename):
        # 例如，从 "unknown_pmcid_123.json" 中提取 123
        match = re.search(r'unknown_pmcid_(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        return float('inf') # 如果格式不匹配，放到最后或进行错误处理
    
    # 3. 对文件列表进行自然数字排序
    json_files_to_process = sorted(json_files_to_process_unsorted, key=sort_key_natural)
    total_files = len(json_files_to_process)
    print(f"Found {total_files} JSON files in '{input_dir}' for GLM cleaning.")

    for i, filename in enumerate(json_files_to_process):
        input_file_path = os.path.join(input_dir, filename)
        # 使用新的输出目录和可能的文件名后缀来区分
        output_filename = os.path.splitext(filename)[0] + "_glm_v3.json" 
        output_file_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_file_path):
            print(f"Skipping file {i+1}/{total_files}: {filename} (output already exists at {output_file_path})")
            continue

        print(f"Processing file {i+1}/{total_files}: {filename}...")
        
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                original_text_units = json.load(f)
        except Exception as e:
            print(f"Error reading or parsing JSON file {input_file_path}: {e}")
            continue

        cleaned_text_units = []
        if not isinstance(original_text_units, list):
            print(f"Warning: Content of {input_file_path} is not a list. Skipping.")
            continue
        segments_total = len(original_text_units)
        for j, unit in enumerate(original_text_units):
            progress_bar_width = 30
            progress = int((j + 1) / segments_total * progress_bar_width)
            progress_str = f"  Cleaning segment {j+1}/{segments_total} for {filename} [{'=' * progress}{' ' * (progress_bar_width - progress)}]"
            print(progress_str, end='\r')

            if isinstance(unit, str) and unit.strip():
                cleaned_unit = clean_text_with_glm_api(unit)
                if cleaned_unit: 
                    cleaned_text_units.append(cleaned_unit)
                if API_CALL_DELAY > 0:
                    time.sleep(API_CALL_DELAY)

        print(" " * 80, end='\r') # 清除进度条打印行
        print(" " * (len(progress_str) + 5), end='\r') # 清除进度条打印行
        if cleaned_text_units:
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_text_units, f, ensure_ascii=False, indent=2)
                print(f"  Successfully saved cleaned text ({len(cleaned_text_units)} units) to {output_file_path}")
            except Exception as e:
                print(f"Error writing cleaned JSON to {output_file_path}: {e}")
        else:
            print(f"  No valid text units remained after cleaning for {filename}.")

if __name__ == '__main__':
    if ZHIPUAI_API_KEY == "YOUR_OWN_ZHIPUAI_API_KEY" or not ZHIPUAI_API_KEY:
        print("错误：请在脚本中设置您的 ZHIPUAI_API_KEY。")
    elif not os.path.isdir(INPUT_JSON_DIR):
        print(f"错误：找不到输入目录 '{INPUT_JSON_DIR}'。请确保路径正确。")
    else:
        print(f"--- Starting GLM Secondary Cleaning (v2 - Quote Fix) ---")
        print(f"Input directory: {INPUT_JSON_DIR}")
        print(f"Output directory: {OUTPUT_CLEANED_JSON_DIR}")
        print(f"Using model: {MODEL_NAME}")
        if API_CALL_DELAY > 0:
            print(f"Delay between API calls: {API_CALL_DELAY} seconds")
        
        process_json_files_with_glm(INPUT_JSON_DIR, OUTPUT_CLEANED_JSON_DIR)
        print(f"--- GLM Secondary Cleaning Finished ---")