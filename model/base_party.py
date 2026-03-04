"""
Server 和 Client 基类 - 定义联邦学习框架的标准接口

所有模型的 Server 和 Client 实现应继承这些基类
"""

import copy
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.nn.functional import sigmoid, binary_cross_entropy
from tqdm import tqdm
import math


class BaseServer(ABC):
    """
    Server 基类 - 服务器端聚合逻辑
    
    每个模型需要实现特定的训练和评估方法
    """
    
    def __init__(self, id: int, d_name: str, num_m: int, total_clients: list,
                 clients: list, evaluate_data: np.ndarray, user_dic: dict, args):
        self.id = id
        self.domain_name = d_name
        self.clients = clients
        self.total_clients = total_clients
        self.num_items = num_m
        self.num_users = len(clients)
        self.args = args
        self.device = args.device
        
        # 用户和物品嵌入
        self.V = torch.randn(num_m, args.embedding_size, device=args.device)
        self.U = torch.randn(self.num_users, args.embedding_size, device=args.device)
        torch.nn.init.uniform_(self.U, a=0., b=1.)
        torch.nn.init.uniform_(self.V, a=0., b=1.)
        
        # 评估数据
        self.evaluate_data = torch.tensor(evaluate_data).to(args.device)
        
        # 域注意力
        self.domain_attention = torch.randn(1, args.num_domain, device=args.device)
        self.user_embedding_with_attention = torch.zeros_like(self.U)
        self.item_embedding_with_attention = torch.zeros_like(self.V)
        
        # 评估指标
        self.lg10 = torch.Tensor([math.log(2) / math.log(i + 2) for i in range(10)]).to(args.device)
        self.lg5 = torch.Tensor([math.log(2) / math.log(i + 2) for i in range(5)]).to(args.device)
        
        self.user_dic = user_dic
        self.mlp = None
        
        # 早停机制
        self.best_hr = 0.0
        self.patience = 5
        self.early_stop_counter = 0
        self.best_model_state = None
    
    @abstractmethod
    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        pass
    
    @abstractmethod
    def train_gnn(self, **kwargs):
        """训练 GNN 模型"""
        pass
    
    def mf_train(self):
        """矩阵分解训练"""
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        
        for bt in tqdm(range(batch_num), desc="MF Training"):
            grads, p = [], []
            item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt+1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]
            
            for it in batch_user:
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue
                map_id = self.user_dic[it][self.domain_name]
                grad, items = self.total_clients[it].train(self.id, map_id, self.U, self.V)
                grads.append(grad)
                item_interact_table[items] += 1
            
            item_interact_table[item_interact_table == 0] = 1
            for it, vl in enumerate(grads):
                u_grad, i_grad = vl[0], vl[1]
                map_id = self.user_dic[batch_user[it]][self.domain_name]
                self.U[map_id] -= u_grad
                self.V -= i_grad / item_interact_table.unsqueeze(1)
    
    def metric_at_k(self, test_predictions: torch.Tensor, k: int, epoch_id: int):
        """计算 HR@K 和 NDCG@K"""
        length = int(len(test_predictions) / 100)
        test_predictions = test_predictions.reshape(length, 100)
        values, indices = torch.topk(test_predictions, k, dim=1, largest=True)
        loc = indices == 99
        hr = torch.sum(loc).item() / length
        if k == 10:
            ndcg = torch.sum(self.lg10 * loc).item() / length
        else:
            ndcg = torch.sum(self.lg5 * loc).item() / length
        return hr, ndcg
    
    def test(self, U: torch.Tensor, V: torch.Tensor, epoch_id: int):
        """通用测试方法"""
        test_data = self.evaluate_data
        with torch.no_grad():
            test_user, test_item = test_data[:, 0], test_data[:, 1]
            test_predictions = sigmoid(torch.sum(torch.multiply(U[test_user], V[test_item]), dim=-1))
            hr_5, ndcg_5 = self.metric_at_k(test_predictions, 5, epoch_id)
            hr_10, ndcg_10 = self.metric_at_k(test_predictions, 10, epoch_id)
            return hr_5, ndcg_5, hr_10, ndcg_10
    
    @abstractmethod
    def test_gnn(self, epoch_id: int):
        """测试 GNN 模型"""
        pass
    
    def test_mf(self, epoch_id: int):
        """测试 MF 模型"""
        return self.test(self.U, self.V, epoch_id)
    
    @abstractmethod
    def kt_stage(self, tf_flag: bool = False, **kwargs):
        """知识转移阶段"""
        pass


class BaseClient(ABC):
    """
    Client 基类 - 客户端训练逻辑
    
    每个模型需要实现特定的训练方法
    """
    
    def __init__(self, id: int, train_data: list, num_m: int, 
                 rating_mean: float, domain_names: list, args):
        self.id = id
        self.rating_mean = rating_mean
        self.train_data = [torch.tensor(train_data[i], device=args.device) 
                          for i in range(args.num_domain)]
        self.items = train_data
        self.knowledge = [[] for _ in range(args.num_domain)]
        self.num_items = num_m
        self.unselected = []
        self.mlp = []
        self.args = args
        self.device = args.device
        self.domain_names = domain_names
        
        # 差分隐私参数
        self.delta = torch.tensor(args.delta, device=args.device)
        self.sensitivity = torch.sqrt(torch.tensor(1, device=args.device))
        self.eps = args.eps
    
    @abstractmethod
    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        pass
    
    @abstractmethod
    def train_gnn(self, **kwargs):
        """训练 GNN 模型"""
        pass
    
    def reset(self, input: torch.Tensor) -> torch.Tensor:
        """重置张量，用于梯度计算"""
        output = torch.clone(input).detach()
        output.requires_grad = True
        output.grad = torch.zeros_like(output)
        return output
    
    @staticmethod
    def sample_negative(data: torch.Tensor, num: int):
        """负采样"""
        neg = torch.randint(0, num, (4 * len(data), 1), device=data.device, 
                           dtype=torch.int64).squeeze()
        rating = torch.cat((torch.ones(len(data), device=data.device),
                           torch.zeros(len(neg), device=data.device)), dim=0)
        neg = torch.cat((data, neg), dim=0)
        return neg, rating
    
    def train(self, domain_id: int, map_id: int, user_embedding: torch.Tensor, 
              item_embedding: torch.Tensor):
        """MF 训练"""
        domain_items, domain_ratings = self.sample_negative(
            self.train_data[domain_id], self.num_items[domain_id])
        item_emb = self.reset(item_embedding)
        user_emb = self.reset(user_embedding[map_id])
        optimizer = torch.optim.Adam([user_emb, item_emb], lr=self.args.lr_mf)
        
        for _ in range(self.args.local_epoch):
            optimizer.zero_grad()
            predict = torch.sum(torch.multiply(user_emb, item_emb[domain_items]), dim=1)
            predict = sigmoid(predict)
            loss = binary_cross_entropy(predict, domain_ratings)
            loss.backward()
            optimizer.step()
        
        grads = [user_embedding[map_id].detach() - user_emb.detach(), 
                 item_embedding.detach() - item_emb.detach()]
        return grads, domain_items
    
    @staticmethod
    def l2_clip(x: torch.Tensor, s: float) -> torch.Tensor:
        """L2 范数裁剪，用于差分隐私"""
        norm = torch.norm(x)
        if norm > s:
            return s * (x / norm)
        else:
            return x
    
    def add_dp_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """添加差分隐私噪声"""
        std = self.sensitivity * torch.sqrt(2 * torch.log(1.25 / self.delta)) * 1 / self.eps
        noise = torch.normal(mean=0, std=std, size=tensor.shape, device=tensor.device)
        return tensor + noise
