import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 使用 ModelScope 作为备选
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件（使用TextLoader避免NLTK依赖）
loader = TextLoader(markdown_path, encoding='utf-8')
docs = loader.load()

# 中文嵌入模型 - 使用 ModelScope 下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5')
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
# llm = ChatOpenAI(
#     model="gpt-4.1-free",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://aihubmix.com/v1"
# )

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 用户查询
question = "文中举了哪些例子？"

# 定义不同的chunk_size和chunk_overlap组合
chunk_configs = [
    (500, 50),
    (1000, 100),
    (2000, 200),
    (4000, 400)
]

# 循环测试不同的配置
for idx, (chunk_size, chunk_overlap) in enumerate(chunk_configs, 1):
    print(f"\n{'='*50}")
    print(f"测试配置 {idx}: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"{'='*50}")
    
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    print(f"生成的chunks数量: {len(chunks)}")
    
    # 构建向量存储
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)
    
    # 在向量存储中查询相关文档
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    answer_content = answer.content
    
    print(f"回答内容:\n{answer_content}\n")
    
    # 保存结果到文件
    output_file = f"result_chunk{chunk_size}_overlap{chunk_overlap}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"配置: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}\n")
        f.write(f"生成的chunks数量: {len(chunks)}\n")
        f.write(f"问题: {question}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"检索到的上下文:\n{docs_content}\n\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"回答:\n{answer_content}\n")
    
    print(f"结果已保存到: {output_file}")

print("\n所有测试完成！")
