import numpy as np
import datetime
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from mlp import MLPEngine
from utils import * # 导入我们在 utils.py 中写好的工具函数

# ==========================================
# 1. 实验参数配置
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='fedcs')
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
parser.add_argument('--dataset', type=str, default='KuaiRec') # 指向 KuaiRec
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
# 处理参数类型
config['use_cuda'] = config['use_cuda'].lower() in ('true', '1', 'yes') if isinstance(config.get('use_cuda'), str) else bool(config.get('use_cuda'))
config['recall_k'] = [int(item) for item in config['recall_k'].split(',')] if len(config['recall_k']) > 1 else [int(config['recall_k'])]
config['server_model_layers'] = [int(item) for item in config['server_model_layers'].split(',')] if len(config['server_model_layers']) > 1 else int(config['server_model_layers'])
config['client_model_layers'] = [int(item) for item in config['client_model_layers'].split(',')] if len(config['client_model_layers']) > 1 else int(config['client_model_layers'])

# ==========================================
# 2. 初始化环境与加载数据集
# ==========================================
path = 'log/'
if not os.path.exists(path): os.makedirs(path)
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
initLogging(os.path.join(path, current_time+'.txt'))
logging.info("🚀 启动基于 [全局热榜+停留时长] 的解耦版联邦推荐引擎...")

# 自动解析相对路径 (无论你在哪执行，都能准确找到父目录下的 data/KuaiRec)
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "data", config['dataset']))
logging.info(f"📂 数据集路径指向: {dataset_dir}")

# 加载数据
warm_df = pd.read_csv(os.path.join(dataset_dir, 'warm_train.csv'))
cold_df = pd.read_csv(os.path.join(dataset_dir, 'cold_train.csv'))
popularity_dict = np.load(os.path.join(dataset_dir, 'popularity.npy'), allow_pickle=True).item()

# 动态设定系统变量
config['num_users'] = max(warm_df['user_id'].max(), cold_df['user_id'].max()) + 1
config['num_items_train'] = max(warm_df['video_id'].max(), cold_df['video_id'].max()) + 1
warm_user_ids = warm_df['user_id'].unique().tolist()
cold_user_ids = cold_df['user_id'].unique().tolist()
dummy_item_content = np.random.rand(config['num_items_train'], config['content_dim']).astype(np.float32)
dummy_item_ids_map = {i: i for i in range(config['num_items_train'])}

# 使用 utils 中的方法剥离测试集
warm_df_train, warm_test_data = build_test_set(warm_df)
cold_df_train, cold_test_data = build_test_set(cold_df)

engine = MLPEngine(config)

# ==========================================
# 3. 时间切片控制与训练循环
# ==========================================
# COLD_START_ROUND = 50 #gpu版测试
COLD_START_ROUND = 1 #cpu临时版测试


cold_recalls_monitor = [] 

for round in range(config['num_round']):
    logging.info('-' * 60)
    logging.info(f'🔄 第 {round} 轮联邦通信开始...')

    # 阶段路由控制 (路由只负责分配数据源，具体逻辑委托给 engine)
    if round < COLD_START_ROUND:
        current_user_ids, current_df = warm_user_ids, warm_df_train
        logging.info(f'🌱 阶段：基座构建期 (仅 {len(current_user_ids)} 名老用户)')
    elif round == COLD_START_ROUND:
        current_user_ids = warm_user_ids + cold_user_ids
        current_df = pd.concat([warm_df_train, cold_df_train])
        logging.info('💥 阶段：[冷启动触发] 注入新用户！时长梯度开始催熟...')
    else:
        current_user_ids = warm_user_ids + cold_user_ids
        current_df = pd.concat([warm_df_train, cold_df_train])
        logging.info(f'🌿 阶段：融合训练期 (共 {len(current_user_ids)} 名用户)')

    # 调用封装好的 utils 采样方法
    all_train_data = kuairec_weighted_sampling(current_df, config['num_negative'], config['num_items_train'])
    
    # 联邦训练核心
    engine.fed_train_a_round(current_user_ids, all_train_data, round, dummy_item_content)

    # 结果评估分离化
    if round >= COLD_START_ROUND:
        logging.info('🎯 监控：冷启动新用户收敛度')
        cold_recall, _, cold_ndcg = engine.fed_evaluate(cold_test_data, dummy_item_content, dummy_item_ids_map)
        logging.info(f"❄️ 新兵 Recall@20 = {cold_recall[1]:.4f} | NDCG@20 = {cold_ndcg[1]:.4f}")
        cold_recalls_monitor.append(cold_recall[1])
    else:
        logging.info('📊 监控：老用户系统指标')
        warm_recall, _, warm_ndcg = engine.fed_evaluate(warm_test_data, dummy_item_content, dummy_item_ids_map)
        logging.info(f"🔥 老兵 Recall@20 = {warm_recall[1]:.4f} | NDCG@20 = {warm_ndcg[1]:.4f}")

logging.info('=' * 60)
logging.info('🎉 实验完成！冷启动用户 Recall@20 爬坡轨迹：')
logging.info(str(cold_recalls_monitor))