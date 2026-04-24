import os
# 配置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\Anaconda\envs\allinrag\Library\share\Tesseract-OCR\tessdata\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'D:\Anaconda\envs\allinrag\Library\share\Tesseract-OCR\tessdata\tessdata'

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
# elements = partition(
#     filename=pdf_path,
#     content_type="application/pdf"
# )

#使用hi_res模式
# elements = partition_pdf(
#     filename=pdf_path,
#     strategy="hi_res",
#     hi_res_model_name= "yolox",
#     languages=["chinese"]
# )

#使用ocr模式
elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    ocr_languages= ["chinese"]
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)