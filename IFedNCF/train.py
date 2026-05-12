import numpy as np
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from mlp import MLPEngine
from utils import *

# ==========================================
# 1. 实验参数配置
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='fedcs-baseline')
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--server_epoch', type=int, default=1)
parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--reg', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr_client', type=float, default=0.5)
parser.add_argument('--lr_server', type=float, default=0.005)
parser.add_argument('--dataset', type=str, default='KuaiRec') 
parser.add_argument('--content_dim', type=int, default=300)
parser.add_argument('--latent_dim', type=int, default=200)
parser.add_argument('--num_negative', type=int, default=5)
parser.add_argument('--server_model_layers', type=str, default='300')
parser.add_argument('--client_model_layers', type=str, default='400, 200')
parser.add_argument('--recall_k', type=str, default='10, 20, 50') 
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()

config = vars(args)
config['use_cuda'] = config['use_cuda'].lower() in ('true', '1', 'yes') if isinstance(config.get('use_cuda'), str) else bool(config.get('use_cuda'))
config['recall_k'] = [int(item) for item in config['recall_k'].split(',')] if len(config['recall_k']) > 1 else [int(config['recall_k'])]
config['server_model_layers'] = [int(item) for item in config['server_model_layers'].split(',')] if len(config['server_model_layers']) > 1 else int(config['server_model_layers'])
config['client_model_layers'] = [int(item) for item in config['client_model_layers'].split(',')] if len(config['client_model_layers']) > 1 else int(config['client_model_layers'])

# ==========================================
# 2. 规范化路径与纯净数据采样函数
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 强制日志保存在 IFedNCF/log 目录下
log_path = os.path.join(current_dir, 'log')
if not os.path.exists(log_path): 
    os.makedirs(log_path)
    
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
log_file_name = os.path.join(log_path, current_time + '_Baseline.txt')
initLogging(log_file_name)
logging.info("🚀 启动 [基线 Baseline] 原版 IFedNCF 联邦推荐引擎...")

def kuairec_baseline_sampling(df, num_negative, total_items):
    """基线专属负采样：不带停留时长，只返回纯粹的 [users, items, ratings] 三元组"""
    all_train_data = {}
    grouped = df.groupby('user_id')
    for user_id, group in grouped:
        pos_items = group['video_id'].tolist()
        users = [user_id] * len(pos_items)
        items = pos_items[:]
        ratings = [1.0] * len(pos_items) # 全置为隐式反馈 1.0

        for _ in range(len(pos_items) * num_negative):
            neg_item = np.random.randint(0, total_items)
            while neg_item in pos_items:
                neg_item = np.random.randint(0, total_items)
            users.append(user_id)
            items.append(neg_item)
            ratings.append(0.0)
            
        all_train_data[user_id] = [users, items, ratings] # 完美适配原版 engine.py
    return all_train_data

def build_test_set(df):
    test_dict = {'uid': [], 'iid': []}
    train_df_list = []
    for uid, group in df.groupby('user_id'):
        if len(group) > 3:
            test_items = group.tail(2)
            train_df_list.append(group.iloc[:-2])
            for iid in test_items['video_id']:
                test_dict['uid'].append(uid)
                test_dict['iid'].append(iid)
        else:
            train_df_list.append(group)
    return pd.concat(train_df_list), pd.DataFrame(test_dict)

dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "data", config['dataset']))
warm_df = pd.read_csv(os.path.join(dataset_dir, 'warm_train.csv'))
cold_df = pd.read_csv(os.path.join(dataset_dir, 'cold_train.csv'))

config['num_users'] = max(warm_df['user_id'].max(), cold_df['user_id'].max()) + 1
config['num_items_train'] = max(warm_df['video_id'].max(), cold_df['video_id'].max()) + 1
warm_user_ids = warm_df['user_id'].unique().tolist()
cold_user_ids = cold_df['user_id'].unique().tolist()
dummy_item_content = np.random.rand(config['num_items_train'], config['content_dim']).astype(np.float32)
dummy_item_ids_map = {i: i for i in range(config['num_items_train'])}

warm_df_train, warm_test_data = build_test_set(warm_df)
cold_df_train, cold_test_data = build_test_set(cold_df)

engine = MLPEngine(config)

# ==========================================
# 3. 时间切片控制与训练循环
# ==========================================
COLD_START_ROUND = 50  
cold_recalls_monitor = [] # 用于图A：冷启动收敛（滴灌步数）
warm_recalls_monitor = [] # 用于图B：全局稳定性（通信轮次）

pbar = tqdm(range(config['num_round']), desc="⏳ 运行 Baseline", unit="轮", colour="blue", dynamic_ncols=True)

for round in pbar:
    logging.info('-' * 60)
    logging.info(f'🔄 第 {round} 轮联邦通信开始...')

    # 【核心魔改：数据滴灌策略】
    if round < COLD_START_ROUND:
        current_user_ids, current_df = warm_user_ids, warm_df_train
        logging.info(f'🌱 阶段：基座构建期')
    else:
        # 计算当前是冷启动的第几步 (1, 2, 3...)
        interaction_step = round - COLD_START_ROUND + 1
        
        # 滴灌魔法：动态截取每个冷用户的前 interaction_step 个交互视频！
        sliced_cold_df = cold_df_train.groupby('user_id').head(interaction_step)
        
        current_user_ids = warm_user_ids + cold_user_ids
        current_df = pd.concat([warm_df_train, sliced_cold_df])
        logging.info(f'💧 阶段：滴灌融合期 (新用户已暴露前 {interaction_step} 个视频)')

    # Mix-IFedNCF 使用带权采样 (IFedNCF 请换成无权采样 kuairec_baseline_sampling)
    # ✅ 修复后：使用基线专属的无权采样函数
    all_train_data = kuairec_baseline_sampling(current_df, config['num_negative'], config['num_items_train'])
    
    engine.fed_train_a_round(current_user_ids, all_train_data, round, dummy_item_content)

    # 评估与记录
    warm_recall, _, warm_ndcg = engine.fed_evaluate(warm_test_data, dummy_item_content, dummy_item_ids_map)
    warm_recalls_monitor.append(warm_recall[1]) # 永远记录老用户的 Recall@20

    if round >= COLD_START_ROUND:
        cold_recall, _, cold_ndcg = engine.fed_evaluate(cold_test_data, dummy_item_content, dummy_item_ids_map)
        cold_recalls_monitor.append(cold_recall[1]) # 记录新用户的 Recall@20
        pbar.set_postfix({"阶段": "新老融合", "新兵": f"{cold_recall[1]:.4f}", "老兵": f"{warm_recall[1]:.4f}"})
    else:
        pbar.set_postfix({"阶段": "基座构建", "老兵": f"{warm_recall[1]:.4f}"})


# ==========================================
# 4. 纯净保存实验数据 (供画图脚本提取)
# ==========================================
np.save(os.path.join(log_path, 'warm_recalls.npy'), warm_recalls_monitor)
if len(cold_recalls_monitor) > 0:
    np.save(os.path.join(log_path, 'cold_recalls.npy'), cold_recalls_monitor)

logging.info('🎉 实验完成！核心评估数组已保存至 log 文件夹，请使用外部画图脚本生成论文插图。')