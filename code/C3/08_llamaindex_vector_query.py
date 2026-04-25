from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置全局嵌入模型
# Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")
Settings.llm = None

from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5')
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_dir,
)
# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
persist_path = "llamaindex_index_store"

storage_context = StorageContext.from_defaults(persist_dir=persist_path)
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=1)
response = query_engine.query("数据")
print(response)

