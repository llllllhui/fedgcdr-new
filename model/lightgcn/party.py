"""
LightGCN 模型的 Server 和 Client 实现

用于 FedGCDR 联邦跨域推荐系统
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
from .model import LightGCN, MLP
from torch.nn.functional import sigmoid, binary_cross_entropy


@SERVER_REGISTRY.register('lightgcn')
class Server(BaseServer):
    """
    LightGCN 服务器端 - 聚合客户端更新并进行知识转移
    """

    def __init__(self, id, d_name, num_m, total_clients, clients,
                 evaluate_data, user_dic, args):
        super().__init__(id, d_name, num_m, total_clients, clients,
                        evaluate_data, user_dic, args)
        # 使用 LightGCN 替换 GAT，减少层数防止过平滑
        self.item_lightgcn = LightGCN(args, args.embedding_size, args.embedding_size,
                                      args.embedding_size, num_layers=1, dropout=0.1)

    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        return self.item_lightgcn

    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, transfer=False, a=None, transfer_vec=None):
        """训练 LightGCN 模型"""
        return self.train_lightgcn(domain_id, user_dic, model_item,
                                   global_user_embedding, global_item_embedding,
                                   transfer, a, transfer_vec)

    def test_gnn(self, epoch_id: int):
        """测试 LightGCN 模型"""
        self.item_lightgcn.eval()
        return self.test(self.user_embedding_with_attention, self.V, epoch_id)

    def train_mlp(self, batch):
        """训练 MLP"""
        s, t = batch * self.args.user_batch, min((batch + 1) * self.args.user_batch, self.num_users)
        selected_clients = [i for i in range(s, t)]
        grads = []
        for it in selected_clients:
            grads.append(self.total_clients[it].train_mlp(self.mlp))
        p = 1 / len(self.total_clients)
        for it in grads:
            for d in range(self.args.num_domain - 1):
                gd = it[d]
                for i, vl in enumerate(self.mlp[d].parameters()):
                    vl.data -= p * gd[i]

    def kt_stage(self, tf_flag=False, round_id=0):
        """
        知识转移阶段

        学习率衰减策略：
        - 知识获取阶段：使用余弦衰减
        - 知识转移阶段：使用固定学习率
        """
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)

        # 学习率衰减策略
        if tf_flag:
            # 知识转移阶段：使用固定学习率
            current_lr_gnn = self.args.lr_gnn
            current_lr_mf = self.args.lr_mf
        else:
            # 知识获取阶段：使用余弦衰减
            total_rounds = self.args.round_gat
            lr_decay = 0.5 * (1 + math.cos(math.pi * round_id / total_rounds))
            current_lr_gnn = self.args.lr_gnn * lr_decay
            current_lr_mf = self.args.lr_mf * lr_decay

        for bt in tqdm(range(batch_num), desc="KT Stage"):
            grads_model, p, grads_embedding, grads_kt = [], [], [], []
            total_item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt+1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]
            no_trans = self.args.user_batch * 1

            for i, it in enumerate(batch_user):
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue

                if tf_flag is False or i >= no_trans:
                    pk, grad_lightgcn, grad_emb, grad_kt = self.total_clients[it].train_lightgcn(
                        self.id, self.user_dic, self.item_lightgcn, self.U, self.V,
                        lr_gnn=current_lr_gnn, lr_mf=current_lr_mf)
                else:
                    pk, grad_lightgcn, grad_emb, grad_kt = self.total_clients[it].knowledge_transfer_lightgcn(
                        self.id, self.mlp, self.user_dic, self.item_lightgcn,
                        self.U, self.V, self.domain_attention,
                        lr_gnn=current_lr_gnn, lr_mf=current_lr_mf)
                    grads_kt.append(grad_kt)

                total_items = grad_emb[3]
                total_item_interact_table[total_items] += 1
                p.append(pk)
                grads_model.append(grad_lightgcn)
                grads_embedding.append(grad_emb)

            p = torch.Tensor(p)
            p = p / torch.sum(p)
            for i, it in enumerate(grads_model):
                if tf_flag and i < no_trans:
                    self.domain_attention.data -= p[i] * grads_kt[i][0]
                    for mid, mlp in enumerate(self.mlp):
                        for pid, para in enumerate(mlp.parameters()):
                            try:
                                para.data -= p[i] * grads_kt[i][mid+1][pid]
                            except:
                                pass
                for j, vl in enumerate(self.item_lightgcn.parameters()):
                    vl.data -= p[i] * it[j]
            total_item_interact_table[total_item_interact_table == 0] = 1
            for grad in grads_embedding:
                uid, u_emb_att, u_emb, total_items, total_grads = grad
                map_id = self.user_dic[uid][self.domain_name]
                self.user_embedding_with_attention[map_id] = u_emb_att
                self.U[map_id] = u_emb
                self.V[total_items] -= total_grads / total_item_interact_table[total_items].unsqueeze(1)


@CLIENT_REGISTRY.register('lightgcn')
class Client(BaseClient):
    """
    LightGCN 客户端 - 本地训练 LightGCN 模型
    """

    def __init__(self, id, train_data, num_m, rating_mean, domain_names, args):
        super().__init__(id, train_data, num_m, rating_mean, domain_names, args)
        self.lightgcn = None

    def get_gnn_model(self):
        """获取 GNN 模型实例"""
        return self.lightgcn

    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, transfer=False, a=None, transfer_vec=None,
                  lr_gnn=None, lr_mf=None):
        """训练 LightGCN 模型"""
        return self.train_lightgcn(domain_id, user_dic, model_item,
                                   global_user_embedding, global_item_embedding,
                                   transfer, a, transfer_vec, lr_gnn, lr_mf)

    def train_lightgcn(self, domain_id, user_dic, model_item, global_user_embedding,
                       global_item_embedding, transfer=False, a=None, transfer_vec=None,
                       lr_gnn=None, lr_mf=None):
        """LightGCN 训练逻辑"""
        if lr_gnn is None:
            lr_gnn = self.args.lr_gnn
        if lr_mf is None:
            lr_mf = self.args.lr_mf

        grads_lightgcn, grad_emb, grad_kt, temp_vec = [], [], [], [0 for _ in range(self.args.num_domain)]
        length = len(self.items[domain_id])
        self.lightgcn = copy.deepcopy(model_item)
        user_embedding = self.reset(global_user_embedding[user_dic[self.id][self.domain_names[domain_id]]])
        item_embedding = self.reset(global_item_embedding)
        paras = [user_embedding, item_embedding] + [para for para in self.lightgcn.parameters()]
        local_a = a
        mlps = None

        if transfer:
            mlps = copy.deepcopy(self.mlp)
            for mlp in mlps:
                paras += [para for para in mlp.parameters()]
            local_a = self.reset(a)

        optimizer = torch.optim.Adam(paras, lr=lr_gnn)
        total_item, ratings = self.sample_negative(self.train_data[domain_id], self.num_items[domain_id])

        for epoch in range(self.args.local_epoch):
            optimizer.zero_grad()
            if transfer and mlps is not None:
                for i in range(self.args.num_domain):
                    temp_vec[i] = mlps[i](transfer_vec[i])
            h_i, intermediate_emb, ls, lm = self.lightgcn(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size),
                          item_embedding[self.items[domain_id]])),
                transfer, local_a, temp_vec)
            user_emb = h_i[0]
            h_i = item_embedding[total_item]
            predict = sigmoid(torch.sum(torch.multiply(user_emb, h_i), dim=1))
            loss = binary_cross_entropy(predict, ratings) + ls + lm
            loss.backward()
            optimizer.step()

        local_para = [para.data for para in self.lightgcn.parameters()]
        global_para = [para.data for para in model_item.parameters()]
        for i in range(len(local_para)):
            grads_lightgcn.append(global_para[i] - local_para[i])

        with torch.no_grad():
            user_emb, self.knowledge[domain_id], ls, lm = self.lightgcn(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size),
                          item_embedding[self.items[domain_id]])),
                transfer, local_a, transfer_vec)

        grad_emb.append(self.id)
        grad_emb.append(user_emb[0].detach())
        grad_emb.append(user_embedding.detach())
        grad_emb.append(total_item)
        grad_emb.append(global_item_embedding[grad_emb[-1]].detach() -
                       item_embedding[grad_emb[-1]].detach())

        if transfer:
            grad_kt.append(a.detach() - local_a.detach())
            for i in range(self.args.num_domain):
                local_para = [para.data for para in mlps[i].parameters()]
                global_para = [para.data for para in self.mlp[i].parameters()]
                para_grad = []
                for pid in range(len(local_para)):
                    para_grad.append(global_para[pid] - local_para[pid])
                grad_kt.append(para_grad)

        return length, grads_lightgcn, grad_emb, grad_kt

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

    def knowledge_transfer_lightgcn(self, domain_id, mlps, user_dic, item_lightgcn,
                                    user_embedding, item_embedding, a,
                                    lr_gnn=None, lr_mf=None):
        """
        LightGCN 知识转移逻辑

        优化策略:
        1. 降低差分隐私噪声影响 (eps 翻倍)
        2. 知识质量门控 (只转移高质量知识)
        """
        transfer_vec = []
        self.mlp = mlps

        # 方案 1: 降低差分隐私噪声影响
        std = self.sensitivity * torch.sqrt(2 * torch.log(1.25 / self.delta)) * 1 / (self.eps * 2)

        # 方案 2: 知识质量阈值
        knowledge_threshold = 0.5

        for j in range(self.args.num_domain):
            if j == domain_id:
                transfer_vec.append(torch.zeros(self.args.embedding_size, device=self.args.device))
            else:
                if len(self.knowledge[j]) == 0:
                    temp_vec = torch.zeros(self.args.embedding_size, device=self.args.device)
                else:
                    temp_vec = Client.l2_clip(
                        torch.tensor(self.knowledge[j][0], device=self.args.device),
                        self.sensitivity)

                    # 知识质量门控
                    knowledge_norm = torch.norm(temp_vec).item()
                    if knowledge_norm < knowledge_threshold:
                        temp_vec = torch.zeros(self.args.embedding_size, device=self.args.device)

                noise = torch.normal(mean=0, std=std,
                                    size=(1, self.args.embedding_size)).to(self.args.device).squeeze()
                if self.args.dp:
                    transfer_vec.append(temp_vec + noise)
                else:
                    transfer_vec.append(temp_vec)

        return self.train_lightgcn(domain_id, user_dic, item_lightgcn, user_embedding,
                                   item_embedding, True, a, transfer_vec, lr_gnn, lr_mf)
