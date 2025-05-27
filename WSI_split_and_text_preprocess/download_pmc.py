import requests
import time
import os
from xml.etree import ElementTree as ET

# --- 配置参数 ---
# 你的NCBI API Key（强烈推荐，如果没获取可以留空字符串""，但请求速度会慢）
API_KEY = "9555c3b79910a2f917c761492ca5e24eae08" # 替换成你的API Key

# 搜索关键词（可以根据需要调整，更精确的关键词能减少不相关的结果）
# 注意：PMC数据库的搜索语法与PubMed类似。
# 示例：
# "pathology" - 搜索包含“病理学”的文章
# "pathology[Title/Abstract]" - 仅在标题和摘要中搜索“病理学”
# "pathology[MeSH Terms]" - 搜索MeSH医学主题词为“病理学”的文章
# "cancer pathology" - 搜索“癌症”和“病理学”都出现的文章
SEARCH_TERM = "pathology[Title/Abstract] AND open access[filter]"

# 你希望下载的文章数量上限
# 注意：esearch一次最多返回100,000个ID，efetch下载需分批进行。
MAX_ARTICLES_TO_DOWNLOAD = 10000

# 下载文件保存目录
DOWNLOAD_DIR = r"F:\dataset\medical_text"

# E-utilities 请求频率控制
# 默认：无API Key时，建议每秒0.3秒（即每秒3次请求）；有API Key时，建议每秒0.1秒（即每秒10次请求）
REQUEST_DELAY = 0.3 if not API_KEY else 0.1 # 每两次请求之间的延时（秒）

# efetch 每次请求下载的文章数量（一次性发送的PMCID数量）
# NCBI建议efetch每次请求不要超过200个ID，这里设置为50以确保稳定
EFETCH_BATCH_SIZE = 50

# --- 创建下载目录 ---
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
print(f"下载目录已创建或已存在: {DOWNLOAD_DIR}")

# --- 步骤 1: 使用esearch查找PMCID ---
print(f"正在使用esearch查找与 '{SEARCH_TERM}' 相关的PMC文章ID...")

pmc_ids = []
esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
esearch_params = {
    "db": "pmc",
    "term": SEARCH_TERM,
    "retmax": MAX_ARTICLES_TO_DOWNLOAD,
    "retmode": "xml",
    "api_key": API_KEY # 如果有API Key则包含
}

try:
    response = requests.get(esearch_url, params=esearch_params)
    response.raise_for_status() # 检查HTTP错误
    root = ET.fromstring(response.content)

    for id_elem in root.findall(".//IdList/Id"):
        pmc_ids.append(id_elem.text)

    print(f"esearch 找到 {len(pmc_ids)} 篇文章ID。")
    if len(pmc_ids) == 0:
        print("没有找到符合条件的文章，请检查搜索关键词或尝试其他关键词。")
        exit()

except requests.exceptions.RequestException as e:
    print(f"esearch 请求失败: {e}")
    print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
    exit()
except ET.ParseError as e:
    print(f"解析esearch响应XML失败: {e}")
    print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
    exit()

# --- 步骤 2: 使用efetch下载文章全文 ---
print(f"开始使用efetch下载 {len(pmc_ids)} 篇文章的XML全文...")

downloaded_count = 0
for i in range(0, len(pmc_ids), EFETCH_BATCH_SIZE):
    batch_ids = pmc_ids[i : i + EFETCH_BATCH_SIZE]
    ids_string = ",".join(batch_ids)

    efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pmc",
        "id": ids_string,
        "rettype": "full", # 请求全文
        "retmode": "xml",  # 请求XML格式
        "api_key": API_KEY # 如果有API Key则包含
    }

    try:
        response = requests.get(efetch_url, params=efetch_params)
        response.raise_for_status()

        # efetch返回的XML可能包含多篇文章，我们需要拆分并保存
        # efetch的返回通常是 <pmc-articleset> 根元素下包含多个 <article> 元素
        batch_root = ET.fromstring(response.content)

        for article_elem in batch_root.findall(".//article"):
            # 尝试获取文章的PMCID作为文件名
            pmcid_element = article_elem.find(".//article-id[@pub-id-type='pmc']")
            article_pmcid = pmcid_element.text if pmcid_element is not None else f"unknown_pmcid_{downloaded_count}"

            file_name = os.path.join(DOWNLOAD_DIR, f"{article_pmcid}.xml")

            # 将单个文章的XML保存到文件
            # ET.tostring默认返回bytes，需要解码
            with open(file_name, "wb") as f:
                f.write(ET.tostring(article_elem, encoding='utf-8', xml_declaration=True))
            
            downloaded_count += 1
            print(f"已下载并保存文章: {article_pmcid}.xml ({downloaded_count}/{len(pmc_ids)})")

    except requests.exceptions.RequestException as e:
        print(f"efetch 请求失败（PMCID批次: {ids_string}）: {e}")
        print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
    except ET.ParseError as e:
        print(f"解析efetch响应XML失败（PMCID批次: {ids_string}）: {e}")
        print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
    except Exception as e:
        print(f"处理文章时发生未知错误（PMCID批次: {ids_string}）: {e}")
        print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")

    time.sleep(REQUEST_DELAY) # 遵守API请求限制

print(f"\n下载完成！总共下载并保存了 {downloaded_count} 篇文章到 '{DOWNLOAD_DIR}' 目录。")
print("下一步你可以开始对这些XML文件进行解析和文本预处理，以供模型训练使用。")