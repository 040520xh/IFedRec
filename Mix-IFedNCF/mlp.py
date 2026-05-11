import torch
from engine import Engine


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.num_items_train = config['num_items_train']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items_train, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        device = self.embedding_user.weight.device
        if not isinstance(item_indices, torch.Tensor):
            item_indices = torch.tensor(item_indices, device=device, dtype=torch.long)
        else:
            item_indices = item_indices.to(device)
        user_idxs = torch.tensor([0] * len(item_indices), device=device, dtype=torch.long)
        user_embedding = self.embedding_user(user_idxs)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def cold_predict(self, item_embedding):
        device = item_embedding.device
        user_idxs = torch.tensor([0] * item_embedding.shape[0], device=device, dtype=torch.long)
        user_embedding = self.embedding_user(user_idxs)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class Server(torch.nn.Module):
    def __init__(self, config):
        super(Server, self).__init__()
        self.config = config
        self.content_dim = config['content_dim']
        self.latent_dim = config['latent_dim']

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['server_model_layers'][:-1], config['server_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.affine_output = torch.nn.Linear(in_features=config['server_model_layers'][-1], out_features=self.latent_dim)
        self.logistic = torch.nn.Tanh()

    def forward(self, item_content):
        vector = item_content
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.client_model = Client(config)
        self.server_model = Server(config)
        if config.get('use_cuda', False) is True and torch.cuda.is_available():
            device_idx = config.get('device_id', 0)
            device = torch.device('cuda:%d' % device_idx)
            self.client_model.to(device)
            self.server_model.to(device)
        super(MLPEngine, self).__init__(config)
