import json
import os
import requests
from bs4 import BeautifulSoup

def download_image(url, save_dir):
    """下载图片并保存到指定目录"""
    if not url.startswith('http'):
        return url  # 如果不是完整的URL，直接返回原地址
        
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 从URL中提取文件名
        filename = os.path.basename(url)
        save_path = os.path.join(save_dir, filename)
        
        # 保存图片
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return os.path.join('images', filename)  # 返回相对路径
    except Exception as e:
        print(f"下载图片失败 {url}: {e}")
        return url

def extract_emojis(html_file, output_file):
    # 创建images目录
    os.makedirs('images', exist_ok=True)
    
    # 读取HTML文件
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 查找所有符合条件的div元素
    emoji_divs = soup.find_all('div', class_='wbpro-iconbed woo-box-flex woo-box-alignCenter woo-box-justifyCenter')
    
    # 提取title和img的src
    emojis = []
    for div in emoji_divs:
        title = div.get('title', '')
        img = div.find('img')
        if img:
            src = img.get('src', '')
            if title and src:  # 只添加同时有title和src的项
                # 下载图片并获取相对路径
                local_src = download_image(src, 'images')
                emojis.append({
                    'title': f'_{title}_',
                    'src': local_src
                })
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(emojis, f, ensure_ascii=False, indent=2)
    
    print(f"成功提取 {len(emojis)} 个表情符号到 {output_file}")

if __name__ == "__main__":
    html_file = 'a.html'
    output_file = 'emojis.json'
    extract_emojis(html_file, output_file)