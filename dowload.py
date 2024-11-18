import os
import requests
from tqdm import tqdm

# VOC 2012 数据集 URL
download_links = [
    "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
    "http://pjreddie.com/media/files/VOCtest_11-May-2012.tar"
]

# 下载目录
download_dir = "./VOC2012"

# 创建目录
if not os.path.exists(download_dir):
    os.makedirs(download_dir)


def download_file(url, dest_path):
    # 获取文件的大小
    response = requests.head(url)
    file_size = int(response.headers.get('Content-Length', 0))

    # 请求文件内容
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            # 使用 tqdm 显示进度条
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=dest_path) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))  # 更新进度条


# 下载文件
for link in download_links:
    file_name = link.split("/")[-1]
    dest_path = os.path.join(download_dir, file_name)
    print(f"开始下载: {file_name}")
    download_file(link, dest_path)
    print(f"{file_name} 下载完成\n")

print("所有文件下载完成！")
