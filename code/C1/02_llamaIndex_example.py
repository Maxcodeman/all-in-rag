import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# hugging face镜像设置，如果国内环境无法使用启用该设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 使用 ModelScope 作为备选
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )

# 使用 DeepSeek
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com",
    is_chat_model=True
)

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 获取嵌入模型
# Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

from modelscope import snapshot_download

model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5')
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_dir
)

#获取数据并切分文档
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

#创建索引
index = VectorStoreIndex.from_documents(docs)

#创建搜索引擎
query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))