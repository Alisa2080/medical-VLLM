# import xml.etree.ElementTree as ET
# import json
# import os
# import re

# def clean_text_from_element(element):
#     """
#     从一个XML元素中提取并清洗文本内容。
#     会移除常见的XML内部标签如xref, ext-link等的文本影响。
#     """
#     if element is None:
#         return ""
    
#     text_parts = []
    
#     if element.text:
#         text_parts.append(element.text)

#     for child in element:
#         # 增加了更多可能包含元数据或非主要内容的标签，这些标签的内容通常不适合作为MLM训练数据
#         if child.tag in [
#             'xref', 'ext-link', 'inline-formula', 'disp-formula', 'label', 
#             'graphic', 'media', 'uri', 'email', 'contrib-id', 'aff', 
#             'author-notes', 'fn-group', 'ack', 'permissions', 'history', 
#             'counts', 'article-id', 'journal-id', 'issn', 'publisher', 
#             'pub-date', 'article-categories', 'issue', 'volume', 'fpage', 'lpage',
#             'copyright-statement', 'copyright-year', 'copyright-holder', 'license-p',
#             'related-article', 'self-uri', 'funding-group', 'award-group', 'funding-statement',
#             'custom-meta-group', 'table', 'thead', 'tbody', 'tr', 'th', 'td', # 跳过表格内部结构标签的递归，表格内容由caption处理
#             'alternatives', # 常用于包裹图片或公式的不同格式
#             'supplementary-material' # 通常补充材料的描述在caption里，这里避免提取其内部复杂结构
#         ]:
#             pass 
#         else:
#             text_parts.append(clean_text_from_element(child))
        
#         if child.tail:
#             text_parts.append(child.tail)
            
#     full_text = "".join(text_parts)
    
#     cleaned_text = full_text.replace('\n', ' ').replace('\t', ' ')
#     cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
#     return cleaned_text

# def extract_text_units_from_xml(xml_file_path):
#     """
#     从单个PMC XML文件中提取所有相关的文本单元（段落、标题等）。
#     现在不再单独提取正文的章节标题。
#     """
#     text_units = []
#     try:
#         tree = ET.parse(xml_file_path)
#         root = tree.getroot()

#         # 1. 文章标题 (仍然保留，因为它通常较长且包含重要信息)
#         article_title_element = root.find(".//front/article-meta/title-group/article-title")
#         if article_title_element is not None:
#             title_text = clean_text_from_element(article_title_element)
#             if title_text:
#                 text_units.append(title_text)
        
#         alt_title_element = root.find(".//front/article-meta/title-group/alt-title")
#         if alt_title_element is not None:
#             alt_title_text = clean_text_from_element(alt_title_element)
#             if alt_title_text and alt_title_text not in text_units:
#                 text_units.append(alt_title_text)

#         # 2. 摘要段落
#         for abstract_element in root.findall(".//front/article-meta/abstract"):
#             abstract_title_element = abstract_element.find("./title")
#             if abstract_title_element is not None:
#                 abstract_title_text = clean_text_from_element(abstract_title_element)
#                 # 摘要的标题如果不是泛泛的"Abstract"等，可以考虑保留
#                 if abstract_title_text and abstract_title_text.lower() not in ["abstract", "summary", "author summary", "background", "methodology/principal findings", "conclusions/significance", "methodology", "conclusions", "results", "introduction", "methods", "discussion"]: # 扩展了摘要中常见但可能重复或短小的标题
#                     text_units.append(abstract_title_text)
            
#             for p_element in abstract_element.findall(".//p"): 
#                 p_text = clean_text_from_element(p_element)
#                 if p_text:
#                     text_units.append(p_text)

#         # 3. 正文章节段落 (不再提取章节标题)
#         body_element = root.find(".//body")
#         if body_element is not None:
#             for p_element_direct_in_body in body_element.findall("./p"): # 处理body下直接的p
#                 p_text = clean_text_from_element(p_element_direct_in_body)
#                 if p_text:
#                     text_units.append(p_text)

#             for sec_element in body_element.findall(".//sec"): 
#                 # ################################################
#                 # ## 不再提取章节标题 (sec_title_element) ##
#                 # ################################################
#                 # sec_title_element = sec_element.find("./title") 
#                 # if sec_title_element is not None:
#                 #     sec_title_text = clean_text_from_element(sec_title_element)
#                 #     if sec_title_text:
#                 #         text_units.append(sec_title_text) # 此行已注释/移除
                
#                 # 只提取章节内的段落
#                 for p_element_in_sec in sec_element.findall("./p"): 
#                     p_text = clean_text_from_element(p_element_in_sec)
#                     if p_text:
#                         text_units.append(p_text)
#                 # 也处理章节下直接的文本节点（如果存在且不在p内，不常见）
#                 # sec_direct_text_parts = []
#                 # if sec_element.text and sec_element.text.strip():
#                 #     sec_direct_text_parts.append(sec_element.text.strip())
#                 # for child in sec_element:
#                 #     if child.tail and child.tail.strip():
#                 #         sec_direct_text_parts.append(child.tail.strip())
#                 # sec_cleaned_direct_text = clean_text_from_element(ET.fromstring(f"<temp>{' '.join(sec_direct_text_parts)}</temp>")) # 重新包装以清洗
#                 # if sec_cleaned_direct_text and not sec_element.findall("./p") and not sec_element.findall("./title"): # 确保不是已被p或title提取的内容
#                 #     text_units.append(sec_cleaned_direct_text)


#         # 4. 图表标题/描述 (caption内的段落和标题) - 这些通常是完整的句子，可以保留
#         caption_elements = []
#         caption_elements.extend(root.findall(".//fig/caption"))
#         caption_elements.extend(root.findall(".//table-wrap/caption"))
        
#         for caption_element in caption_elements:
#             caption_title_element = caption_element.find("./title")
#             if caption_title_element is not None:
#                 caption_title_text = clean_text_from_element(caption_title_element)
#                 if caption_title_text:
#                     text_units.append(caption_title_text)
            
#             for p_element_in_caption in caption_element.findall("./p"):
#                 cap_text = clean_text_from_element(p_element_in_caption)
#                 if cap_text:
#                     text_units.append(cap_text)
            
#             if not caption_element.findall("./title") and not caption_element.findall("./p"):
#                 caption_direct_text_cleaned = clean_text_from_element(caption_element) # 清洗整个caption的内容
#                 if caption_direct_text_cleaned:
#                      text_units.append(caption_direct_text_cleaned)

#         # 5. 表格脚注
#         for fn_element in root.findall(".//table-wrap/table-wrap-foot//fn"):
#             found_p_in_fn = False
#             for p_element_in_fn in fn_element.findall("./p"):
#                 fn_text = clean_text_from_element(p_element_in_fn)
#                 if fn_text:
#                     text_units.append(fn_text)
#                     found_p_in_fn = True
#             if not found_p_in_fn:
#                 fn_direct_text_cleaned = clean_text_from_element(fn_element)
#                 if fn_direct_text_cleaned:
#                     text_units.append(fn_direct_text_cleaned)
        
#         # (可选) 补充材料的标题和描述
#         for sup_material_element in root.findall(".//supplementary-material"):
#             sup_label_element = sup_material_element.find("./label")
#             if sup_label_element is not None:
#                 sup_label_text = clean_text_from_element(sup_label_element)
#                 if sup_label_text: # 补充材料的label通常比较短，类似标题
#                     text_units.append(sup_label_text)

#             sup_caption_element = sup_material_element.find("./caption")
#             if sup_caption_element is not None:
#                 # caption下可能有title和p
#                 sup_caption_title_element = sup_caption_element.find("./title")
#                 if sup_caption_title_element is not None:
#                     sup_caption_title_text = clean_text_from_element(sup_caption_title_element)
#                     if sup_caption_title_text:
#                          text_units.append(sup_caption_title_text)
#                 for p_element_in_sup_caption in sup_caption_element.findall("./p"):
#                     p_text_sup_cap = clean_text_from_element(p_element_in_sup_caption)
#                     if p_text_sup_cap:
#                         text_units.append(p_text_sup_cap)


#         # 移除空字符串和去重（保持顺序）
#         seen = set()
#         unique_text_units = []
#         for unit in text_units:
#             if unit and unit not in seen:
#                 # 额外的过滤：去除过短的文本单元，比如少于 N 个词的
#                 # 这里的N可以根据需要调整，例如 N=3 或 N=5
#                 if len(unit.split()) >= 3: # 例如，至少需要3个词
#                     unique_text_units.append(unit)
#                 seen.add(unit)
#         return unique_text_units

#     except ET.ParseError as e:
#         print(f"Error parsing XML file {xml_file_path}: {e}")
#         return []
#     except Exception as e:
#         print(f"An unexpected error occurred with file {xml_file_path}: {e}")
#         return []

# # process_pmc_xml_files 和 if __name__ == '__main__': 部分保持不变
# def process_pmc_xml_files(input_dir, output_dir):
#     """
#     处理输入目录中的所有PMC XML文件，并将提取的文本保存到输出目录的JSON文件中。
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     processed_files = 0
#     error_files = 0

#     for filename in os.listdir(input_dir):
#         if filename.endswith(".xml"):
#             xml_file_path = os.path.join(input_dir, filename)
#             print(f"Processing {xml_file_path}...")

#             text_units = extract_text_units_from_xml(xml_file_path)

#             if text_units:
#                 json_filename = os.path.splitext(filename)[0] + ".json"
#                 json_file_path = os.path.join(output_dir, json_filename)

#                 with open(json_file_path, 'w', encoding='utf-8') as f:
#                     json.dump(text_units, f, ensure_ascii=False, indent=2) # indent=2 减小文件大小
#                 print(f"Saved {len(text_units)} text units to {json_file_path}")
#                 processed_files +=1
#             else:
#                 print(f"No text units extracted (or all too short) from {xml_file_path} or error occurred.")
#                 error_files +=1
    
#     print(f"\n--- Processing Summary ---")
#     print(f"Total XML files processed: {processed_files + error_files}")
#     print(f"Successfully generated JSON files: {processed_files}")
#     print(f"Files with errors or no extractable text: {error_files}")


# if __name__ == '__main__':
#     xml_input_directory = r'F:\dataset\medical_text_XLM' 
#     json_output_directory = r'F:\dataset\medical_text_json'
    
#     print(f"--- Running processing ---")
#     process_pmc_xml_files(xml_input_directory, json_output_directory)
#     print(f"--- Processing finished ---")
#     print(f"Please check the '{json_output_directory}' for JSON output.")

import xml.etree.ElementTree as ET
import json
import os
import re

# 全局跳过标签列表，用于 clean_text_from_element 函数
# 这些标签的 *直接* 文本内容和其子孙的文本内容将被忽略，但它们的 .tail 文本仍会被处理
SKIP_TAGS_FOR_CLEAN_TEXT = [
    'xref',             # 交叉引用 (e.g., [1], Fig 1) - 主要目标是移除其内容
    'ext-link',         # 外部链接
    'inline-formula',   # 行内公式
    'disp-formula',     # 块级公式
    'label',            # 通常是图、表、公式的标签 (e.g., "Figure 1.", "(1)") - 标题/说明由caption提取
    'graphic',          # 图像本身
    'media',            # 多媒体对象
    'uri',              # URI链接文本
    'email',            # 电子邮件地址
    'contrib-id',       # 作者ID
    'aff',              # 单位信息
    'author-notes',     # 作者注释 (如通讯作者信息，利益冲突等)
    'fn-group',         # 一组脚注，单个fn由特定逻辑处理
    'ack',              # 致谢部分，如果不想作为正文可以保留在此
    'permissions',      # 版权许可信息
    'history',          # 文章接收/修改日期历史
    'counts',           # 图表计数
    'article-id',       # 文章ID (PMID, DOI等)
    'journal-id',       # 期刊ID
    'issn',             # ISSN
    'publisher',        # 出版商
    'pub-date',         # 发表日期
    'article-categories',# 文章分类
    'issue', 'volume', 'fpage', 'lpage', # 期刊卷号、页码等
    'copyright-statement', 'copyright-year', 'copyright-holder', 'license-p', # 版权信息
    'related-article',  # 相关文章链接
    'self-uri',         # 指向本文其他格式的链接 (如PDF)
    'funding-group', 'award-group', 'funding-statement', #基金信息
    'custom-meta-group',# 自定义元数据
    'table', 'thead', 'tbody', 'tr', 'th', 'td', # 表格的结构和内容，表格标题由caption提取
    'fig-group',        # 图组
    'alternatives',     # 通常用于包裹公式或图片的不同表示形式
    'supplementary-material', # 补充材料的元数据，其caption可能被单独提取
    'inline-graphic',   # 行内小图
    'private-char',     # 私有字符
    'def-list', 'term', # 定义列表和术语，如果其<p>内容重要，extract_text_units_from_xml应单独处理
    # 'list-item',      # list-item 是结构性标签，它的子<p>等应该被处理，所以不应跳过list-item本身
    'tex-math', 'mml:math', # TeX和MathML数学公式标记
    'processing-meta',  # XML处理元信息
    'journal-meta',     # 期刊元数据
    'kwd-group',        # 关键词组
    'notes',            # 通常包含作者贡献等非主体散文内容
    # 'app', 'app-group',# 附录的结构标签，其下的<p>等应由extract_text_units_from_xml针对性提取
    'back',             # 文章末尾材料的容器标签
    ET.Comment          # XML注释节点
]


def clean_text_from_element(element):
    """
    从一个XML元素中递归提取并清洗文本内容。
    - 确保不同来源的文本片段正确地用空格拼接。
    - 移除空的或仅包含标点的引用残留方括号。
    """
    if element is None:
        return ""

    # 用于收集来自 element.text, child_elements_text, child.tail 的文本片段
    text_fragments = []

    # 1. 处理元素自身的起始文本 (element.text)
    if element.text:
        text_fragments.append(element.text)

    # 2. 遍历所有子元素
    for child in element:
        # a. 如果子标签在全局跳过列表里，则不处理其内部文本，但要处理其 .tail
        #    同时处理 ET.Comment 和 MathML (mml:) 标签
        if child.tag in SKIP_TAGS_FOR_CLEAN_TEXT or \
           child.tag == ET.Comment or \
           (isinstance(child.tag, str) and child.tag.startswith("mml:")):
            if child.tail:
                text_fragments.append(child.tail)
            continue # 跳过此子元素的内部文本，继续下一个兄弟元素

        # b. 递归获取子元素的内部文本
        child_internal_text = clean_text_from_element(child)
        if child_internal_text: # 确保子元素确实返回了文本
            text_fragments.append(child_internal_text)
        
        # c. 处理子元素的尾部文本 (child.tail)
        if child.tail:
            text_fragments.append(child.tail)
            
    # 3. 拼接所有文本片段
    #    - 先对每个片段进行strip()，去除可能由XML带来的不必要的头尾空格。
    #    - 用单个空格连接所有非空的、strip后的片段。
    #    - 这一步是解决文本粘连的关键。
    assembled_text = ' '.join(fragment.strip() for fragment in text_fragments if fragment and fragment.strip())

    # 4. 后续清理
    #    a. 移除引用标记残留：如 [], [,] 或者内部只有空格和逗号/分号的方括号
    #       替换为一个空格，后续会被规范化。
    cleaned_text = re.sub(r'\s*\[\s*([,;\s]*)\s*\]', ' ', assembled_text)
    
    #    b. 规范化所有空白字符：将换行、制表符等变为空格，并将连续空格压缩为单个空格。
    #       最后再 strip() 一次，去除整个结果的头尾空格。
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# extract_text_units_from_xml 和其他函数保持不变
def extract_text_units_from_xml(xml_file_path):
    """
    从单个PMC XML文件中提取所有相关的文本单元（段落、标题等）。
    现在不再单独提取正文的章节标题。
    """
    text_units = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 1. 文章标题 (仍然保留，因为它通常较长且包含重要信息)
        article_title_element = root.find(".//front/article-meta/title-group/article-title")
        if article_title_element is not None:
            title_text = clean_text_from_element(article_title_element)
            if title_text:
                text_units.append(title_text)
        
        alt_title_element = root.find(".//front/article-meta/title-group/alt-title")
        if alt_title_element is not None:
            alt_title_text = clean_text_from_element(alt_title_element)
            if alt_title_text and alt_title_text not in text_units:
                text_units.append(alt_title_text)

        # 2. 摘要段落
        for abstract_element in root.findall(".//front/article-meta/abstract"):
            abstract_title_element = abstract_element.find("./title")
            if abstract_title_element is not None:
                abstract_title_text = clean_text_from_element(abstract_title_element)
                if abstract_title_text and abstract_title_text.lower() not in ["abstract", "summary", "author summary", "background", "methodology/principal findings", "conclusions/significance", "methodology", "conclusions", "results", "introduction", "methods", "discussion"]:
                    text_units.append(abstract_title_text)
            
            for p_element in abstract_element.findall(".//p"): 
                p_text = clean_text_from_element(p_element)
                if p_text:
                    text_units.append(p_text)

        # 3. 正文章节段落 (不再提取章节标题)
        body_element = root.find(".//body")
        if body_element is not None:
            for p_element_direct_in_body in body_element.findall("./p"): 
                p_text = clean_text_from_element(p_element_direct_in_body)
                if p_text: # 确保该段落不属于下面 sec 中会被再次提取的段落
                    # 简单的检查方法是看这个p的父元素是否就是body
                    # 但ElementTree标准库不直接提供获取父元素的方法，
                    # 这里的去重逻辑主要依赖后续的 `seen` 集合。
                    text_units.append(p_text)

            for sec_element in body_element.findall(".//sec"):                 
                for p_element_in_sec in sec_element.findall("./p"): 
                    p_text = clean_text_from_element(p_element_in_sec)
                    if p_text:
                        text_units.append(p_text)

        # 4. 图表标题/描述 (caption内的段落和标题)
        caption_elements = []
        caption_elements.extend(root.findall(".//fig/caption"))
        caption_elements.extend(root.findall(".//table-wrap/caption"))
        
        for caption_element in caption_elements:
            caption_title_element = caption_element.find("./title")
            if caption_title_element is not None:
                caption_title_text = clean_text_from_element(caption_title_element)
                if caption_title_text:
                    text_units.append(caption_title_text)
            
            for p_element_in_caption in caption_element.findall("./p"):
                cap_text = clean_text_from_element(p_element_in_caption)
                if cap_text:
                    text_units.append(cap_text)
            
            if not caption_element.findall("./title") and not caption_element.findall("./p"):
                caption_direct_text_cleaned = clean_text_from_element(caption_element) 
                if caption_direct_text_cleaned:
                     text_units.append(caption_direct_text_cleaned)

        # 5. 表格脚注
        for fn_element in root.findall(".//table-wrap/table-wrap-foot//fn"):
            found_p_in_fn = False
            for p_element_in_fn in fn_element.findall("./p"):
                fn_text = clean_text_from_element(p_element_in_fn)
                if fn_text:
                    text_units.append(fn_text)
                    found_p_in_fn = True
            if not found_p_in_fn:
                fn_direct_text_cleaned = clean_text_from_element(fn_element)
                if fn_direct_text_cleaned: # 确保fn本身在没有p时，其文本内容被提取
                    text_units.append(fn_direct_text_cleaned)
        
        # 6. 附录中的段落 (直接在 <app> 下的 <p>)
        for app_element in root.findall(".//back/app-group/app"):
            app_title_element = app_element.find("./title") # 附录标题
            if app_title_element is not None:
                app_title_text = clean_text_from_element(app_title_element)
                if app_title_text: # 附录标题通常较短，但可以考虑保留
                    text_units.append(app_title_text)
            for p_element_in_app in app_element.findall("./p"): # 附录正文段落
                app_p_text = clean_text_from_element(p_element_in_app)
                if app_p_text:
                    text_units.append(app_p_text)
        
        # 7. (可选) 补充材料的标题和描述
        for sup_material_element in root.findall(".//supplementary-material"):
            sup_label_element = sup_material_element.find("./label")
            if sup_label_element is not None:
                sup_label_text = clean_text_from_element(sup_label_element)
                if sup_label_text: 
                    text_units.append(sup_label_text)

            sup_caption_element = sup_material_element.find("./caption")
            if sup_caption_element is not None:
                sup_caption_title_element = sup_caption_element.find("./title")
                if sup_caption_title_element is not None:
                    sup_caption_title_text = clean_text_from_element(sup_caption_title_element)
                    if sup_caption_title_text:
                         text_units.append(sup_caption_title_text)
                for p_element_in_sup_caption in sup_caption_element.findall("./p"):
                    p_text_sup_cap = clean_text_from_element(p_element_in_sup_caption)
                    if p_text_sup_cap:
                        text_units.append(p_text_sup_cap)


        # 移除空字符串和去重（保持顺序），并过滤过短文本
        seen = set()
        unique_text_units = []
        for unit in text_units:
            if unit and unit not in seen:
                if len(unit.split()) >= 3: 
                    unique_text_units.append(unit)
                seen.add(unit)
        return unique_text_units

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred with file {xml_file_path}: {e}")
        return []

def process_pmc_xml_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_files = 0
    error_files = 0
    total_files = 0

    xml_files = [f for f in os.listdir(input_dir) if f.endswith(".xml")]
    total_files = len(xml_files)
    
    print(f"Found {total_files} XML files to process.")

    for i, filename in enumerate(xml_files):
        xml_file_path = os.path.join(input_dir, filename)
        print(f"Processing file {i+1}/{total_files}: {xml_file_path}...")

        text_units = extract_text_units_from_xml(xml_file_path)

        if text_units:
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_file_path = os.path.join(output_dir, json_filename)

            try:
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(text_units, f, ensure_ascii=False, indent=2)
                # print(f"Saved {len(text_units)} text units to {json_file_path}") #减少打印信息
                processed_files +=1
            except IOError as e:
                print(f"IOError writing JSON for {xml_file_path}: {e}")
                error_files +=1
        else:
            # print(f"No text units extracted (or all too short) from {xml_file_path} or error occurred.") #减少打印信息
            # 如果extract_text_units_from_xml返回空列表是因为解析错误，错误已打印
            # 如果是因为没有内容或内容太短，也算一种处理完成但无输出的情况
            error_files +=1 # Or count separately if needed
    
    print(f"\n--- Processing Summary ---")
    print(f"Total XML files found: {total_files}")
    print(f"Successfully generated JSON files: {processed_files}")
    print(f"Files with errors or no extractable/valid text: {error_files}")


if __name__ == '__main__':
    # 请确保这里的路径是您实际的路径
    xml_input_directory = r'F:\dataset\medical_text_XLM' 
    json_output_directory = r'F:\dataset\medical_text_json'
    
    # 简单检查路径是否存在
    if not os.path.isdir(xml_input_directory):
        print(f"Error: Input directory not found: {xml_input_directory}")
        exit()

    print(f"--- Running processing ---")
    print(f"Input directory: {xml_input_directory}")
    print(f"Output directory: {json_output_directory}")
    process_pmc_xml_files(xml_input_directory, json_output_directory)
    print(f"--- Processing finished ---")
    print(f"Please check the '{json_output_directory}' for JSON output.")