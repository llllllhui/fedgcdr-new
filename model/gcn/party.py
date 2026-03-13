"""
GCN server and client implementation.
"""

import copy
import math
import os
import sys

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, sigmoid
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_party import BaseServer, BaseClient
from registry import SERVER_REGISTRY, CLIENT_REGISTRY
from .model import GCN


@SERVER_REGISTRY.register('gcn')
class Server(BaseServer):
    def __init__(self, id, d_name, num_m, total_clients, clients,
                 evaluate_data, user_dic, args):
        super().__init__(id, d_name, num_m, total_clients, clients,
                         evaluate_data, user_dic, args)
        self.gnn_model = GCN(args, args.embedding_size, args.embedding_size, args.embedding_size)

    def get_gnn_model(self):
        return self.gnn_model

    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, transfer=False, a=None, transfer_vec=None):
        return self.train_gcn(domain_id, user_dic, model_item, global_user_embedding,
                              global_item_embedding, transfer, a, transfer_vec)

    def train_gcn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, transfer=False, a=None, transfer_vec=None):
        return self.gnn_model

    def test_gnn(self, epoch_id: int):
        self.gnn_model.eval()
        return self.test(self.user_embedding_with_attention, self.V, epoch_id)

    def kt_stage(self, tf_flag=False, round_id=0):
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        current_lr = self.args.lr_gnn

        for bt in tqdm(range(batch_num), desc="GCN KT Stage"):
            grads_model, p, grads_embedding = [], [], []
            total_item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt + 1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]

            for it in batch_user:
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue

                length, grad_gnn, grad_emb, grad_kt = self.total_clients[it].train_gnn(
                    self.id, self.user_dic, self.gnn_model, self.U, self.V, lr=current_lr)

                total_items = grad_emb[3]
                total_item_interact_table[total_items] += 1
                p.append(length)
                grads_model.append(grad_gnn)
                grads_embedding.append(grad_emb)

            if not p:
                continue

            p = torch.Tensor(p)
            p = p / torch.sum(p)
            for i, grad_set in enumerate(grads_model):
                for j, parameter in enumerate(self.gnn_model.parameters()):
                    parameter.data -= p[i] * grad_set[j]

            total_item_interact_table[total_item_interact_table == 0] = 1
            for grad in grads_embedding:
                uid, u_emb_att, u_emb, total_items, total_grads = grad
                map_id = self.user_dic[uid][self.domain_name]
                self.user_embedding_with_attention[map_id] = u_emb_att
                self.U[map_id] = u_emb
                self.V[total_items] -= total_grads / total_item_interact_table[total_items].unsqueeze(1)


@CLIENT_REGISTRY.register('gcn')
class Client(BaseClient):
    def __init__(self, id, train_data, num_m, rating_mean, domain_names, args):
        super().__init__(id, train_data, num_m, rating_mean, domain_names, args)
        self.gnn_model = None

    def get_gnn_model(self):
        return self.gnn_model

    def train_gnn(self, domain_id, user_dic, model_item, global_user_embedding,
                  global_item_embedding, lr=None):
        if lr is None:
            lr = self.args.lr_gnn

        grads_gnn, grad_emb = [], []
        length = len(self.items[domain_id])
        self.gnn_model = copy.deepcopy(model_item)

        user_embedding = self.reset(
            global_user_embedding[user_dic[self.id][self.domain_names[domain_id]]])
        item_embedding = self.reset(global_item_embedding)
        parameters = [user_embedding, item_embedding] + list(self.gnn_model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr)

        total_item, ratings = self.sample_negative(
            self.train_data[domain_id], self.num_items[domain_id])

        for _ in range(self.args.local_epoch):
            optimizer.zero_grad()
            h_i, intermediate_emb, ls, lm = self.gnn_model(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size),
                           item_embedding[self.items[domain_id]])))

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

        return length, grads_gnn, grad_emb, []

    def train_mlp(self, mlps):
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
