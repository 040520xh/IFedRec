import torch
from utils import *
import numpy as np
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader


class Engine(object):
    """Meta Engine for training & evaluating NCF model"""

    def __init__(self, config):
        self.config = config  
        if config.get('use_cuda', False) is True and torch.cuda.is_available():
            device_idx = config.get('device_id', 0)
            self.device = torch.device('cuda:%d' % device_idx)
        else:
            self.device = torch.device('cpu')
            
        self.server_opt = torch.optim.Adam(self.server_model.parameters(), lr=config.get('lr_server', 0.001),
                                           weight_decay=config.get('l2_regularization', 0.0))
        self.server_model_param = {}
        self.client_model_params = {}
        
        # 阻断 PyTorch 自动求平均，以便逐个样本乘上停留时长权重
        self.client_crit = torch.nn.BCELoss(reduction='none') 
        self.server_crit = torch.nn.MSELoss()

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # 接收并应用时长权重
        _, items, ratings, weights = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
        
        ratings = ratings.float()
        weights = weights.float()
        
        items, ratings, weights = items.to(self.device), ratings.to(self.device), weights.to(self.device)

        optimizer = optimizers[0]
        optimizer.zero_grad()

        ratings_pred = model_client(items)
        
        # 计算加权 Loss
        base_loss = self.client_crit(ratings_pred.view(-1), ratings)
        weighted_loss = base_loss * weights
        loss = weighted_loss.mean()
        
        # 【避坑】：只在真实文本存在 (reg > 0) 时去计算无用的 global_item_rep
        if self.config.get('reg', 0.0) > 0.0:
            reg_item_embedding = copy.deepcopy(self.server_model_param['global_item_rep'])
            reg_item_embedding = reg_item_embedding.to(self.device)
            regularization_term = compute_regularization(model_client, reg_item_embedding)
            loss += self.config['reg'] * regularization_term
        
        loss.backward()
        optimizer.step()
        return model_client

    def aggregate_clients_params(self, round_user_params, item_content):
        """IFed 原版解耦聚合：只聚合物品特征"""
        t = 0
        for user in round_user_params.keys():
            user_params = round_user_params[user]
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)

        if self.config.get('reg', 0.0) > 0.0:
            if isinstance(item_content, torch.Tensor):
                item_content = item_content.clone().detach()
            else:
                item_content = torch.tensor(item_content)
            target = self.server_model_param['embedding_item.weight'].data
            item_content = item_content.to(self.device)
            target = target.to(self.device)
            self.server_model.train()
            for epoch in range(self.config.get('server_epoch', 1)):
                self.server_opt.zero_grad()
                logit_rep = self.server_model(item_content)
                loss = self.server_crit(logit_rep, target)
                loss.backward()
                self.server_opt.step()

            self.server_model.eval()
            with torch.no_grad():
                global_item_rep = self.server_model(item_content)
            self.server_model_param['global_item_rep'] = global_item_rep

    def fed_train_a_round(self, user_ids, all_train_data, round_id, item_content):
        """捍卫版：纯正解耦联邦架构，猛兽级本地学习率"""
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(len(user_ids) * self.config['clients_sample_ratio'])
            num_participants = min(num_participants, len(user_ids))
            participants = np.random.choice(user_ids, num_participants, replace=False)
        else:
            participants = np.random.choice(user_ids, self.config['clients_sample_num'], replace=False)

        if round_id == 0:
            self.server_model_param = {}
            self.server_model_param['embedding_item.weight'] = copy.deepcopy(self.client_model.embedding_item.weight.data).cpu()
            
            if self.config.get('reg', 0.0) > 0.0:
                item_content_t = torch.tensor(item_content).to(self.device)
                self.server_model.eval()
                with torch.no_grad():
                    global_item_rep = self.server_model(item_content_t)
                self.server_model_param['global_item_rep'] = global_item_rep

        round_participant_params = {}

        for user in participants:
            model_client = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())
            
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(self.device)
            
            if round_id != 0 and 'embedding_item.weight' in self.server_model_param:
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).to(self.device)
                        
            model_client.load_state_dict(user_param_dict)

            # 【致胜关键】：本地学习率 0.01 激活 MLP
            optimizer = torch.optim.Adam(model_client.parameters(), lr=0.01)
            optimizers = [optimizer]

            user_train_data = all_train_data[user]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()

            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client = self.fed_train_single_batch(model_client, batch, optimizers)
            
            client_param = model_client.state_dict()
            
            if user not in self.client_model_params:
                self.client_model_params[user] = {}
            for key in client_param.keys():
                if key != 'embedding_item.weight':
                    self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()
            
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(client_param['embedding_item.weight'].data).cpu()
                    
        self.aggregate_clients_params(round_participant_params, item_content)

    def fed_evaluate(self, evaluate_data, item_content, item_ids_map):
        """匹配解耦逻辑的测试函数"""
        import torch
        import copy

        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {} 
        all_item_indices = torch.arange(self.config['num_items_train']).to(self.device)

        for user in user_ids:
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())
            
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(self.device)
            
            if 'embedding_item.weight' in self.server_model_param:
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).to(self.device)
            
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            
            with torch.no_grad():
                preds = user_model(all_item_indices)
                user_item_preds[user] = preds.view(-1).cpu()

        recall, precision, ndcg = compute_metrics(
            evaluate_data,            
            user_item_preds,          
            item_ids_map,             
            self.config['recall_k']   
        )
        return recall, precision, ndcg