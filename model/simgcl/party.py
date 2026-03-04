"""
SimGCL 模型的 Server 和 Client 实现示例
"""

import copy
import torch
import numpy as np
from tqdm import tqdm
import math
import sys
import os

# 导入基类和注册表
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_party import BaseServer, BaseClient
from registry import SERVER_REGISTRY, CLIENT_REGISTRY
from .model import SimGCL, MLP
from torch.nn.functional import sigmoid, binary_cross_entropy


@SERVER_REGISTRY.register('simgcl')
class Server(BaseServer):
    """
    SimGCL 服务器端 - 聚合客户端更新
    
    这是一个示例实现
    """
    
    def __init__(self, id, d_name, num_m, total_clients, clients,
                 evaluate_data, user_dic, args):
        super().__init__(id, d_name, num_m, total_clients, clients,
                        evaluate_data, user_dic, args)
        self.gnn_model = SimGCL(args, args.embedding_size,
                                args.embedding_size, args.embedding_size,
                                eps=0.1, temp=0.2)
    
    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        return self.gnn_model

    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, transfer=False, a=None, transfer_vec=None):
        """训练 SimGCL 模型"""
        return self.train_simgcl(domain_id, user_dic, model_item,
                                 global_user_embedding, global_item_embedding,
                                 transfer, a, transfer_vec)

    def test_gnn(self, epoch_id: int):
        """测试 SimGCL 模型"""
        self.gnn_model.eval()
        return self.test(self.user_embedding_with_attention, self.V, epoch_id)

    def kt_stage(self, tf_flag=False, round_id=0):
        """
        知识转移阶段
        
        SimGCL 版本，使用对比学习增强
        """
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        
        # 学习率衰减
        if tf_flag:
            current_lr = self.args.lr_gnn
        else:
            total_rounds = self.args.round_gat
            lr_decay = 0.5 * (1 + math.cos(math.pi * round_id / total_rounds))
            current_lr = self.args.lr_gnn * lr_decay
        
        for bt in tqdm(range(batch_num), desc="SimGCL KT Stage"):
            grads_model, p, grads_embedding = [], [], []
            total_item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt+1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]
            
            for i, it in enumerate(batch_user):
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue
                
                length, grad_gnn, grad_emb, grad_kt = self.total_clients[it].train_gnn(
                    self.id, self.user_dic, self.gnn_model,
                    self.U, self.V, lr=current_lr)

                pk = length  # 兼容接口
                
                total_items = grad_emb[3]
                total_item_interact_table[total_items] += 1
                p.append(pk)
                grads_model.append(grad_gnn)
                grads_embedding.append(grad_emb)
            
            # 聚合梯度
            p = torch.Tensor(p)
            p = p / torch.sum(p)
            for i, it in enumerate(grads_model):
                for j, vl in enumerate(self.gnn_model.parameters()):
                    vl.data -= p[i] * it[j]
            
            # 更新嵌入
            total_item_interact_table[total_item_interact_table == 0] = 1
            for grad in grads_embedding:
                uid, u_emb_att, u_emb, total_items, total_grads = grad
                map_id = self.user_dic[uid][self.domain_name]
                self.user_embedding_with_attention[map_id] = u_emb_att
                self.U[map_id] = u_emb
                self.V[total_items] -= total_grads / total_item_interact_table[total_items].unsqueeze(1)


@CLIENT_REGISTRY.register('simgcl')
class Client(BaseClient):
    """
    SimGCL 客户端 - 本地训练 SimGCL 模型
    """
    
    def __init__(self, id, train_data, num_m, rating_mean, domain_names, args):
        super().__init__(id, train_data, num_m, rating_mean, domain_names, args)
        self.gnn_model = None
    
    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        return self.gnn_model
    
    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, lr=None):
        """
        SimGCL 训练逻辑 - 包含对比学习
        """
        if lr is None:
            lr = self.args.lr_gnn
        
        grads_gnn, grad_emb = [], []
        length = len(self.items[domain_id])
        
        # 复制全局模型
        self.gnn_model = copy.deepcopy(model_item)
        
        # 重置嵌入
        user_embedding = self.reset(
            global_user_embedding[user_dic[self.id][self.domain_names[domain_id]]])
        item_embedding = self.reset(global_item_embedding)
        
        # 优化器
        paras = [user_embedding, item_embedding] + list(self.gnn_model.parameters())
        optimizer = torch.optim.Adam(paras, lr=lr)
        
        # 负采样
        total_item, ratings = self.sample_negative(
            self.train_data[domain_id], self.num_items[domain_id])
        
        # 本地训练
        for epoch in range(self.args.local_epoch):
            optimizer.zero_grad()
            
            # 前向传播 (包含对比学习)
            h_i, intermediate_emb, cl_loss, lm = self.gnn_model(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size),
                          item_embedding[self.items[domain_id]])))
            
            user_emb = h_i[0]
            h_i = item_embedding[total_item]
            
            # 计算总损失 (BCE + 对比学习 + 知识转移)
            predict = sigmoid(torch.sum(torch.multiply(user_emb, h_i), dim=1))
            loss = binary_cross_entropy(predict, ratings) + cl_loss + 0.01 * lm
            loss.backward()
            optimizer.step()
        
        # 计算梯度
        local_para = [para.data for para in self.gnn_model.parameters()]
        global_para = [para.data for para in model_item.parameters()]
        for i in range(len(local_para)):
            grads_gnn.append(global_para[i] - local_para[i])
        
        # 准备返回
        with torch.no_grad():
            user_emb, self.knowledge[domain_id], _, _ = self.gnn_model(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size),
                          item_embedding[self.items[domain_id]])))
        
        grad_emb.append(self.id)
        grad_emb.append(user_emb[0].detach())
        grad_emb.append(user_embedding.detach())
        grad_emb.append(total_item)
        grad_emb.append(global_item_embedding[grad_emb[-1]].detach() -
                       item_embedding[grad_emb[-1]].detach())
        
        return length, grads_gnn, grad_emb, []
    
    def train_mlp(self, mlps):
        """训练 MLP"""
        self.mlp = mlps
        grads = []
        for d in range(self.args.num_domain - 1):
            if len(self.knowledge[d]) == 0:
                continue
            temp_vec = torch.tensor(self.knowledge[d][0], device=self.args.device)
            temp_vec = self.l2_clip(temp_vec, self.sensitivity)
            mlps[d].zero_grad()
            output = mlps[d](temp_vec)
            loss = torch.norm(output - temp_vec) ** 2
            loss.backward()
            grad = [p.grad.data for p in mlps[d].parameters()]
            grads.append(grad)
        return grads
