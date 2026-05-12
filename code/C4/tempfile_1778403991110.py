import requests

# 测试是否能处理 Brotli 压缩
url = "https://api.bilibili.com/x/web-interface/view?bvid=BV1Bo4y1A7FU"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

try:
    response = requests.get(url, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"Content-Encoding: {response.headers.get('Content-Encoding', '无')}")
    print(f"响应内容长度: {len(response.text)} 字符")
    print("✅ Brotli 解码成功！")
except Exception as e:
    print(f"❌ 错误: {e}")
