import torch
from utils import *
import numpy as np
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.client_model and self.server_model!
    """

    def __init__(self, config):
        self.config = config  # model configuration
        # determine device: prefer CUDA only if requested and available
        if config.get('use_cuda', False) is True and torch.cuda.is_available():
            device_idx = config.get('device_id', 0)
            self.device = torch.device('cuda:%d' % device_idx)
        else:
            self.device = torch.device('cpu')
        self.server_opt = torch.optim.Adam(self.server_model.parameters(), lr=config['lr_server'],
                                           weight_decay=config['l2_regularization'])
        self.server_model_param = {}
        self.client_model_params = {}
        self.client_crit = torch.nn.BCELoss()
        self.server_crit = torch.nn.MSELoss()

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        reg_item_embedding = copy.deepcopy(self.server_model_param['global_item_rep'])

        items, ratings = items.to(self.device), ratings.to(self.device)
        reg_item_embedding = reg_item_embedding.to(self.device)

        optimizer = optimizers[0]
        optimizer.zero_grad()
        ratings_pred = model_client(items)
        loss = self.client_crit(ratings_pred.view(-1), ratings)
        regularization_term = compute_regularization(model_client, reg_item_embedding)
        loss += self.config['reg'] * regularization_term
        loss.backward()
        optimizer.step()
        return model_client

    def aggregate_clients_params(self, round_user_params, item_content):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():
            # load a user's parameters.
            user_params = round_user_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)

      # train the item representation learning module.
        if isinstance(item_content, torch.Tensor):
            item_content = item_content.clone().detach()
        else:
            item_content = torch.tensor(item_content)
        target = self.server_model_param['embedding_item.weight'].data
        item_content = item_content.to(self.device)
        target = target.to(self.device)
        self.server_model.train()
        for epoch in range(self.config['server_epoch']):
            self.server_opt.zero_grad()
            logit_rep = self.server_model(item_content)
            loss = self.server_crit(logit_rep, target)
            loss.backward()
            self.server_opt.step()

        # store the global item representation learned by server model.
        self.server_model.eval()
        with torch.no_grad():
            global_item_rep = self.server_model(item_content)
        self.server_model_param['global_item_rep'] = global_item_rep


    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # 解包 Dataset 传来的 4 个返回值，包含 weights
        _, items, ratings, weights = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
        
        ratings = ratings.float()
        weights = weights.float()
        
        items, ratings, weights = items.to(self.device), ratings.to(self.device), weights.to(self.device)

        optimizer = optimizers[0]
        optimizer.zero_grad()

        ratings_pred = model_client(items)
        
        # 加权 Loss 计算 (时长赋权核心)
        base_loss = self.client_crit(ratings_pred.view(-1), ratings)
        weighted_loss = base_loss * weights
        loss = weighted_loss.mean()
        
        # 【完美避坑】：只有 reg > 0 时（即真实文本拉力存在时），才去索要并计算 global_item_rep
        if self.config.get('reg', 0.0) > 0.0:
            reg_item_embedding = copy.deepcopy(self.server_model_param['global_item_rep'])
            reg_item_embedding = reg_item_embedding.to(self.device)
            regularization_term = compute_regularization(model_client, reg_item_embedding)
            loss += self.config['reg'] * regularization_term
        
        loss.backward()
        optimizer.step()
        return model_client


    def fed_evaluate(self, evaluate_data, item_content, item_ids_map):
        """捍卫版：匹配解耦逻辑的测试函数（不再强加 Server 的 MLP）"""
        import torch
        import copy

        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {} 
        all_item_indices = torch.arange(self.config['num_items_train']).to(self.device)

        for user in user_ids:
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())
            
            # 1. 完全加载该用户本地强大的个性化网络
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(self.device)
            
            # 2. 拼接最新的全局 Item Embedding
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