import pandas as pd
import numpy as np
import os
import random
import urllib.request
import zipfile

# 设置随机种子，保证实验结果的可复现性
random.seed(42)
np.random.seed(42)

def download_kuairec(output_dir):
    """
    自动从官方 Zenodo 链接下载 KuaiRec 数据集并解压
    """
    url = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
    zip_path = os.path.join(output_dir, "KuaiRec.zip")
    
    # 【已修复】：加上了 KuaiRec 2.0 这个层级
    extracted_csv_path = os.path.join(output_dir, "KuaiRec 2.0", "data", "small_matrix.csv")

    if os.path.exists(extracted_csv_path):
        print(f"📦 检测到本地已存在 KuaiRec 数据集: {extracted_csv_path}")
        return extracted_csv_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"⬇️ 正在从官方节点下载 KuaiRec 数据集 (压缩包约 432MB)...")
    print(f"🔗 下载链接: {url}")
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            percent = min(percent, 100.0)
            print(f"\r🚀 下载进度: {percent:.1f}% ({readsofar/(1024*1024):.1f} MB / {totalsize/(1024*1024):.1f} MB)", end='')

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\n✅ 下载完成！")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return None

    print("📂 正在解压缩文件...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("✅ 解压完成！")

    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("🗑️ 已清理原始压缩包。")

    return extracted_csv_path


def preprocess_kuairec(raw_data_path, output_dir):
    """
    KuaiRec 数据集预处理函数 (融合停留时长赋权)
    """
    print("\n" + "="*40)
    print("1. 读取 KuaiRec 原始数据...")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {raw_data_path}。")
        return

    # 提取我们需要的核心列
    df = df[['user_id', 'video_id', 'play_duration', 'video_duration']]

    print("2. 计算停留时长权重 (Weight w)...")
    # 过滤掉 video_duration 为 0 或异常的脏数据，防止除以 0
    df = df[df['video_duration'] > 0]
    
    # 设定权重截断上限 MAX_W
    MAX_W = 3.0 
    
    # 公式: w = min(play_duration / video_duration, MAX_W)
    df['weight'] = (df['play_duration'] / df['video_duration']).clip(upper=MAX_W)

    # 为了兼容 IFedRec 默认的隐式反馈逻辑（有交互即为 1）
    df['rating'] = 1.0

    print("3. 划分 Warm (80%) 和 Cold (20%) 集合...")
    unique_users = df['user_id'].unique()
    unique_items = df['video_id'].unique()

    # 随机打乱用户和物品序列
    np.random.shuffle(unique_users)
    np.random.shuffle(unique_items)

    # 计算 80% 的切分点
    user_split_idx = int(len(unique_users) * 0.8)
    item_split_idx = int(len(unique_items) * 0.8)

    # 划分老用户和老物品
    warm_users = set(unique_users[:user_split_idx])
    warm_items = set(unique_items[:item_split_idx])

    # Warm 组
    warm_df = df[df['user_id'].isin(warm_users) & df['video_id'].isin(warm_items)].copy()

    # Cold 组：专门用来做冷启动测试的 20% 新用户
    cold_users = set(unique_users[user_split_idx:])
    cold_df = df[df['user_id'].isin(cold_users)].copy()

    print("4. 生成全局热榜字典 (Popularity Dict)...")
    # 统计 Warm 组中，各个老物品被交互的总次数
    item_counts = warm_df['video_id'].value_counts().to_dict()

    # 将热度进行归一化到 0~1 之间
    max_count = max(item_counts.values()) if item_counts else 1
    popularity_dict = {item: count / max_count for item, count in item_counts.items()}

    print("5. 正在保存处理后的数据...")
    # 确保存储产物的目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cols_to_save = ['user_id', 'video_id', 'rating', 'weight']
    
    warm_df[cols_to_save].to_csv(os.path.join(output_dir, 'warm_train.csv'), index=False)
    cold_df[cols_to_save].to_csv(os.path.join(output_dir, 'cold_train.csv'), index=False)
    
    # 将字典保存为 numpy 格式
    np.save(os.path.join(output_dir, 'popularity.npy'), popularity_dict)

    print("\n🎉 数据预处理全部完成！")
    print(f"📊 Warm 组 (老用户) 交互数: {len(warm_df)}")
    print(f"❄️ Cold 组 (新用户) 交互数: {len(cold_df)}")
    print(f"🔥 全局热榜记录物品数: {len(popularity_dict)}")
    print(f"📁 你的训练数据已保存在: {os.path.abspath(output_dir)}")
    print("="*40)


if __name__ == "__main__":
    # 配置根目录
    BASE_DATA_DIR = "data/KuaiRec"
    
    # 第一步：自动下载并解压获取 CSV 路径
    raw_csv_path = download_kuairec(BASE_DATA_DIR)
    
    # 第二步：如果下载/检测成功，开始预处理
    if raw_csv_path:
        preprocess_kuairec(raw_csv_path, BASE_DATA_DIR)