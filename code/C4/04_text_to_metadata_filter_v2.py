import os
from langchain_deepseek import ChatDeepSeek 
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging
import requests
import re
import json

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
    "https://www.bilibili.com/video/BV1z14y1c78y"
]

def extract_bvid(url):
    """从 URL 中提取 BV 号"""
    match = re.search(r'BV(\w+)', url)
    if match:
        return match.group(0)
    return None

def fetch_bilibili_video_info(bvid):
    """使用 requests 直接获取 Bilibili 视频信息"""
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.bilibili.com/"
    }
    
    try:
        print(f"正在获取视频信息: {bvid}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 0:
            video_data = data['data']
            info = {
                'title': video_data.get('title', '未知标题'),
                'author': video_data.get('owner', {}).get('name', '未知作者'),
                'view_count': video_data.get('stat', {}).get('view', 0),
                'duration': video_data.get('duration', 0),
                'description': video_data.get('desc', '') or video_data.get('title', ''),
                'bvid': bvid,
                'cid': video_data.get('cid', 0)
            }
            print(f"✅ 成功获取: {info['title']}")
            return info
        else:
            print(f"❌ API 返回错误: {data.get('message', '未知错误')}")
            return None
    except Exception as e:
        print(f"❌ 获取视频信息失败: {type(e).__name__}: {e}")
        return None

bili = []

# 尝试方法 1: 使用 BiliBiliLoader
print("=" * 60)
print("方法 1: 尝试使用 BiliBiliLoader...")
print("=" * 60)
try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    for doc in docs:
        original = doc.metadata
        
        # 提取基本元数据字段
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        doc.metadata = metadata
        bili.append(doc)
    
    print(f"✅ BiliBiliLoader 成功加载 {len(bili)} 条数据\n")
    
except Exception as e:
    print(f"❌ BiliBiliLoader 失败: {type(e).__name__}: {str(e)}\n")

# 如果方法 1 失败，使用方法 2: 直接调用 API
if not bili:
    print("=" * 60)
    print("方法 2: 使用备用方案（直接调用 Bilibili API）...")
    print("=" * 60)
    
    for url in video_urls:
        bvid = extract_bvid(url)
        if bvid:
            video_info = fetch_bilibili_video_info(bvid)
            
            if video_info:
                doc = Document(
                    page_content=video_info['description'],
                    metadata={
                        'title': video_info['title'],
                        'author': video_info['author'],
                        'source': video_info['bvid'],
                        'view_count': video_info['view_count'],
                        'length': video_info['duration']
                    }
                )
                bili.append(doc)
    
    print(f"\n备用方案共获取 {len(bili)} 条数据")

if not bili:
    print("\n❌ 所有方法都失败了，程序退出")
    exit()

print(f"\n✅ 总共加载 {len(bili)} 个视频")
for i, doc in enumerate(bili, 1):
    print(f"{i}. {doc.metadata['title']} (时长: {doc.metadata['length']}秒)")

# 2. 创建向量存储
print("\n--> 正在创建向量存储...")
embed_model = HuggingFaceEmbeddings(model_name=r"D:\modelscope\models\BAAI\bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)
print("✅ 向量存储创建完成")

# 3. 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

# 4. 初始化LLM客户端
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 5. 获取所有文档用于排序
all_documents = vectorstore.similarity_search("", k=len(bili)) 

# 6. 执行查询示例
queries = [
    "时间最短的视频",
    "播放量最高的视频",
    "时长大于600秒的视频",
    "视频观看次数最多的前两个视频"
]

for query in queries:
    print(f"\n--- 原始查询: '{query}' ---")

    # 使用大模型将自然语言转换为排序或过滤指令
    prompt = f"""你是一个智能助手，请将用户的问题转换成一个JSON指令，用于检索和排序视频。

你需要识别用户的意图：
1. 如果是排序查询，生成包含 sort_by 和 order 的指令
2. 如果是过滤查询，生成包含 filter 条件的指令

支持的排序字段：
- 'view_count' (观看次数)
- 'length' (时长)

支持的排序方向：
- 'asc' (升序/最短/最少)
- 'desc' (降序/最长/最多)

支持的过滤操作符：
- 'gt' (大于)
- 'lt' (小于)
- 'gte' (大于等于)
- 'lte' (小于等于)
- 'eq' (等于)

示例:
- '时间最短的视频' -> {{"type": "sort", "sort_by": "length", "order": "asc"}}
- '播放量最高的视频' -> {{"type": "sort", "sort_by": "view_count", "order": "desc"}}
- '时长大于600秒的视频' -> {{"type": "filter", "field": "length", "operator": "gt", "value": 600}}
- '观看次数最多的前两个视频' -> {{"type": "sort", "sort_by": "view_count", "order": "desc", "limit": 2}}

请根据以下问题生成JSON指令:
原始问题: "{query}"

JSON指令:"""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        instruction_str = response.choices[0].message.content
        instruction = json.loads(instruction_str)
        print(f"--- 生成的指令: {instruction} ---")

        instruction_type = instruction.get('type')

        if instruction_type == 'sort':
            # 处理排序查询
            sort_by = instruction.get('sort_by')
            order = instruction.get('order')
            limit = instruction.get('limit', 1)

            if sort_by in ['length', 'view_count'] and order in ['asc', 'desc']:
                reverse_order = (order == 'desc')
                sorted_docs = sorted(all_documents, key=lambda doc: doc.metadata.get(sort_by, 0), reverse=reverse_order)
                
                # 限制返回数量
                sorted_docs = sorted_docs[:limit]
                
                print(f"\n找到 {len(sorted_docs)} 个匹配的视频:")
                for idx, doc in enumerate(sorted_docs, 1):
                    title = doc.metadata.get('title', '未知标题')
                    author = doc.metadata.get('author', '未知作者')
                    view_count = doc.metadata.get('view_count', '未知')
                    length = doc.metadata.get('length', '未知')
                    print(f"\n{idx}. 标题: {title}")
                    print(f"   作者: {author}")
                    print(f"   观看次数: {view_count}")
                    print(f"   时长: {length}秒")
                print("="*50)
            else:
                print("生成的排序指令无效，无法执行排序")

        elif instruction_type == 'filter':
            # 处理过滤查询
            field = instruction.get('field')
            operator = instruction.get('operator')
            value = instruction.get('value')

            if field in ['length', 'view_count'] and operator in ['gt', 'lt', 'gte', 'lte', 'eq']:
                filtered_docs = []
                for doc in all_documents:
                    field_value = doc.metadata.get(field, 0)
                    
                    # 根据操作符进行过滤
                    if operator == 'gt' and field_value > value:
                        filtered_docs.append(doc)
                    elif operator == 'lt' and field_value < value:
                        filtered_docs.append(doc)
                    elif operator == 'gte' and field_value >= value:
                        filtered_docs.append(doc)
                    elif operator == 'lte' and field_value <= value:
                        filtered_docs.append(doc)
                    elif operator == 'eq' and field_value == value:
                        filtered_docs.append(doc)
                
                print(f"\n找到 {len(filtered_docs)} 个匹配的视频:")
                for idx, doc in enumerate(filtered_docs, 1):
                    title = doc.metadata.get('title', '未知标题')
                    author = doc.metadata.get('author', '未知作者')
                    view_count = doc.metadata.get('view_count', '未知')
                    length = doc.metadata.get('length', '未知')
                    print(f"\n{idx}. 标题: {title}")
                    print(f"   作者: {author}")
                    print(f"   观看次数: {view_count}")
                    print(f"   时长: {length}秒")
                print("="*50)
            else:
                print("生成的过滤指令无效，无法执行过滤")
        else:
            print("未知的指令类型")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"解析或执行指令失败: {e}")
