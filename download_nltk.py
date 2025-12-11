import nltk

print("开始检查并下载NLTK数据包，请稍候...")

# 将所有需要的数据包都放在这个列表里
packages = [
    'punkt',
    'averaged_perceptron_tagger',
    'wordnet',
    'stopwords',
    'averaged_perceptron_tagger_eng' # <-- 新增的、本次报错所需的数据包
]

for package_id in packages:
    try:
        # 这是一个更通用的检查方法
        nltk.data.find(f'tokenizers/{package_id}' if package_id == 'punkt' else f'corpora/{package_id}' if package_id in ['wordnet', 'stopwords'] else f'taggers/{package_id}')
        print(f"✅ 数据包 '{package_id}' 已存在。")
    except LookupError:
        print(f" R'正在下载 '{package_id}'...")
        nltk.download(package_id)

print("\n✅ 所有需要的数据包都已准备就绪！")