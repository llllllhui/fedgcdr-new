"""
GraphSAGE 模型的 Server 和 Client 实现示例

这是一个模板，展示如何实现新模型的 Server/Client
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
from .model import GraphSAGE, MLP
from torch.nn.functional import sigmoid, binary_cross_entropy


@SERVER_REGISTRY.register('graphsage')
class Server(BaseServer):
    """
    GraphSAGE 服务器端 - 聚合客户端更新
    
    这是一个示例实现，展示了如何为新模型创建 Server
    """
    
    def __init__(self, id, d_name, num_m, total_clients, clients, 
                 evaluate_data, user_dic, args):
        super().__init__(id, d_name, num_m, total_clients, clients, 
                        evaluate_data, user_dic, args)
        self.gnn_model = GraphSAGE(args, args.embedding_size, 
                                   args.embedding_size, args.embedding_size)
    
    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        return self.gnn_model

    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, transfer=False, a=None, transfer_vec=None):
        """训练 GraphSAGE 模型"""
        return self.train_graphsage(domain_id, user_dic, model_item,
                                   global_user_embedding, global_item_embedding,
                                   transfer, a, transfer_vec)

    def test_gnn(self, epoch_id: int):
        """测试 GraphSAGE 模型"""
        self.gnn_model.eval()
        return self.test(self.user_embedding_with_attention, self.V, epoch_id)

    def kt_stage(self, tf_flag=False, round_id=0):
        """知识转移阶段"""
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)

        current_lr = self.args.lr_gnn
        no_trans = self.args.user_batch * 1
        grads_kt = []

        for bt in tqdm(range(batch_num), desc="GraphSAGE KT Stage"):
            grads_model, p, grads_embedding = [], [], []
            total_item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt + 1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]

            for i, it in enumerate(batch_user):
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue

                if tf_flag is False or i >= no_trans:
                    length, grad_gnn, grad_emb, _ = self.total_clients[it].train_gnn(
                        self.id, self.user_dic, self.gnn_model,
                        self.U, self.V, lr=current_lr)
                else:
                    length, grad_gnn, grad_emb, grad_kt_single = self.total_clients[it].knowledge_transfer_graphsage(
                        self.id, self.mlp, self.user_dic, self.gnn_model,
                        self.U, self.V, self.domain_attention, lr=current_lr)
                    grads_kt.append(grad_kt_single)

                total_items = grad_emb[3]
                total_item_interact_table[total_items] += 1
                p.append(length)
                grads_model.append(grad_gnn)
                grads_embedding.append(grad_emb)

            if not p:
                continue

            p = torch.Tensor(p)
            p = p / torch.sum(p)
            for i, it in enumerate(grads_model):
                if tf_flag and i < no_trans:
                    self.domain_attention.data -= p[i] * grads_kt[i][0]
                    for mid, mlp in enumerate(self.mlp):
                        for pid, para in enumerate(mlp.parameters()):
                            try:
                                para.data -= p[i] * grads_kt[i][mid + 1][pid]
                            except Exception:
                                pass
                for j, vl in enumerate(self.gnn_model.parameters()):
                    vl.data -= p[i] * it[j]

            total_item_interact_table[total_item_interact_table == 0] = 1
            for grad in grads_embedding:
                uid, u_emb_att, u_emb, total_items, total_grads = grad
                map_id = self.user_dic[uid][self.domain_name]
                self.user_embedding_with_attention[map_id] = u_emb_att
                self.U[map_id] = u_emb
                self.V[total_items] -= total_grads / total_item_interact_table[total_items].unsqueeze(1)


@CLIENT_REGISTRY.register('graphsage')
class Client(BaseClient):
    """
    GraphSAGE 客户端 - 本地训练 GraphSAGE 模型
    
    这是一个示例实现，展示了如何为新模型创建 Client
    """
    
    def __init__(self, id, train_data, num_m, rating_mean, domain_names, args):
        super().__init__(id, train_data, num_m, rating_mean, domain_names, args)
        self.gnn_model = None
    
    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        return self.gnn_model
    
    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, lr=None, transfer=False, a=None, transfer_vec=None):
        if lr is None:
            lr = self.args.lr_gnn

        grads_gnn, grad_emb, grad_kt = [], [], []
        length = len(self.items[domain_id])
        self.gnn_model = copy.deepcopy(model_item)

        user_embedding = self.reset(
            global_user_embedding[user_dic[self.id][self.domain_names[domain_id]]])
        item_embedding = self.reset(global_item_embedding)

        paras = [user_embedding, item_embedding] + list(self.gnn_model.parameters())
        temp_vec = [0 for _ in range(self.args.num_domain)]
        local_a = a
        mlps = None

        if transfer:
            mlps = copy.deepcopy(self.mlp)
            for mlp in mlps:
                paras += [p for p in mlp.parameters()]
            local_a = self.reset(a)

        optimizer = torch.optim.Adam(paras, lr=lr)

        total_item, ratings = self.sample_negative(
            self.train_data[domain_id], self.num_items[domain_id])

        for epoch in range(self.args.local_epoch):
            optimizer.zero_grad()
            if transfer and mlps is not None:
                for i in range(self.args.num_domain):
                    temp_vec[i] = mlps[i](transfer_vec[i])

            h_i, intermediate_emb, ls, lm = self.gnn_model(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size),
                          item_embedding[self.items[domain_id]])),
                transfer, local_a, temp_vec)

            user_emb = h_i[0]
            h_i = item_embedding[total_item]

            predict = sigmoid(torch.sum(torch.multiply(user_emb, h_i), dim=1))
            loss = binary_cross_entropy(predict, ratings) + 0.01 * ls + 0.01 * lm
            loss.backward()
            optimizer.step()

        local_para = [para.data for para in self.gnn_model.parameters()]
        global_para = [para.data for para in model_item.parameters()]
        for i in range(len(local_para)):
            grads_gnn.append(global_para[i] - local_para[i])

        with torch.no_grad():
            user_emb, self.knowledge[domain_id], ls, lm = self.gnn_model(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size),
                          item_embedding[self.items[domain_id]])))

        grad_emb.append(self.id)
        grad_emb.append(user_emb[0].detach())
        grad_emb.append(user_embedding.detach())
        grad_emb.append(total_item)
        grad_emb.append(global_item_embedding[grad_emb[-1]].detach() -
                       item_embedding[grad_emb[-1]].detach())

        if transfer:
            grad_kt.append(a.detach() - local_a.detach())
            for i in range(self.args.num_domain):
                lp = [p.data for p in mlps[i].parameters()]
                gp = [p.data for p in self.mlp[i].parameters()]
                grad_kt.append([g - l for g, l in zip(gp, lp)])

        return length, grads_gnn, grad_emb, grad_kt

    def knowledge_transfer_graphsage(self, domain_id, mlps, user_dic, model_item,
                                      user_embedding, item_embedding, a, lr=None):
        """GraphSAGE 知识转移：加 DP 噪声后调用 train_gnn(transfer=True)"""
        transfer_vec = []
        self.mlp = mlps
        std = self.sensitivity * torch.sqrt(2 * torch.log(1.25 / self.delta)) * 1 / (self.eps * 2)

        for j in range(self.args.num_domain):
            if j == domain_id:
                transfer_vec.append(torch.zeros(self.args.embedding_size, device=self.args.device))
            else:
                if len(self.knowledge[j]) == 0:
                    temp_vec = torch.zeros(self.args.embedding_size, device=self.args.device)
                else:
                    temp_vec = self.l2_clip(
                        torch.tensor(self.knowledge[j][0], device=self.args.device),
                        self.sensitivity)
                    if torch.norm(temp_vec).item() < 0.5:
                        temp_vec = torch.zeros(self.args.embedding_size, device=self.args.device)
                noise = torch.normal(mean=0, std=std,
                                    size=(1, self.args.embedding_size)).to(self.args.device).squeeze()
                if self.args.dp:
                    transfer_vec.append(temp_vec + noise)
                else:
                    transfer_vec.append(temp_vec)

        return self.train_gnn(domain_id, user_dic, model_item, user_embedding,
                              item_embedding, lr=lr, transfer=True, a=a, transfer_vec=transfer_vec)
    
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
