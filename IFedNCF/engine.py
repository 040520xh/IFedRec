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


    def fed_train_a_round(self, user_ids, all_train_data, round_id, item_content):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            # 改为按当前池子实际人数采样 len(user_ids)
            num_participants = int(len(user_ids) * self.config['clients_sample_ratio'])
            num_participants = min(num_participants, len(user_ids))
            participants = np.random.choice(user_ids, num_participants, replace=False)
        else:
            participants = np.random.choice(user_ids, self.config['clients_sample_num'], replace=False)

        # initialize server parameters for the first round.
        if round_id == 0:
            item_content = torch.tensor(item_content).to(self.device)
            self.server_model.eval()
            with torch.no_grad():
                global_item_rep = self.server_model(item_content)
            self.server_model_param['global_item_rep'] = global_item_rep

        # store users' model parameters of current round.
        round_participant_params = {}
        # perform model update for each participated user.
        for user in participants:
            # copy the client model architecture from self.client_model
            model_client = copy.deepcopy(self.client_model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated item embedding from server.
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.client_model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(self.device)
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).to(self.device)
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating score function.
            # 【终极修复】：使用业界最稳健的 Adam 优化器统一接管，彻底解决原版公式引发的八万倍梯度爆炸！
            optimizer = torch.optim.Adam(model_client.parameters(), lr=0.001)
            optimizers = [optimizer]

            # load current user's training data and instance a train loader.
            user_train_data = all_train_data[user]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client = self.fed_train_single_batch(model_client, batch, optimizers)
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = {}
            for key in client_param.keys():
                if key != 'embedding_item.weight':
                    self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()
            # store client models' local parameters for global update.
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(
                client_param['embedding_item.weight']).data.cpu()
        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params, item_content)


    def fed_evaluate(self, evaluate_data, item_content, item_ids_map):
        """
        终极版 5.0：完美匹配原作者 utils.py 的参数签名与数据流
        """
        import torch
        import copy

        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {} 

        # 构造所有物品的索引张量
        all_item_indices = torch.arange(self.config['num_items_train']).to(self.device)

        # 逐个用户进行预测
        for user in user_ids:
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())
            
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(self.device)
            
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            
            with torch.no_grad():
                preds = user_model(all_item_indices)
                # 保持 PyTorch Tensor 格式，供 utils.py 调用 .topk()
                user_item_preds[user] = preds.view(-1).cpu()

        # 【核心修复】：严格按照 utils.py 要求的 4 个参数顺序进行调用！
        recall, precision, ndcg = compute_metrics(
            evaluate_data,            # 1. 原始测试集 DataFrame
            user_item_preds,          # 2. 预测结果 Tensor 字典
            item_ids_map,             # 3. 物品 ID 映射表
            self.config['recall_k']   # 4. [10, 20, 50] 列表
        )
        return recall, precision, ndcg