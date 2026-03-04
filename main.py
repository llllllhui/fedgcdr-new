import random
import numpy as np
import pandas as pd
import torch
import os
import json
import importlib
import subprocess

import math
import argparse
import warnings
import datetime
import utility
from checkpoint import CheckpointManager, restore_from_checkpoint, restore_target_domain
from model import get_server_class, get_client_class, get_model_class, list_all_models


def get_git_commit_hash():
    """获取当前git commit hash"""
    try:
        # 获取短版本hash（前7位）
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return 'unknown'
    except Exception:
        return 'unknown'


def get_model_display_name(gnn_type):
    """根据gnn_type获取模型显示名称"""
    model_names = {
        'gat': 'GAT',
        'lightgcn': 'LightGCN',
        'graphsage': 'GraphSAGE',
        'simgcl': 'SimGCL'
    }
    return model_names.get(gnn_type, gnn_type.upper())

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='args for fedgcdr')
parser.add_argument('--dataset', choices=['amazon', 'douban'], default='amazon')
parser.add_argument('--round_gat', type=int, default=30)
parser.add_argument('--round_ft', type=int, default=60)
parser.add_argument('--num_domain', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--target_domain', type=int, default=1)
parser.add_argument('--lr_mf', type=float, default=0.005)
parser.add_argument('--lr_gnn', type=float, default=0.01,
                    help='GNN模型统一学习率 (gat/lightgcn/graphsage/simgcl共用)')
parser.add_argument('--embedding_size', type=int, default=16)
parser.add_argument('--local_epoch', type=int, default=3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--user_batch', type=int, default=16)
parser.add_argument('--model', type=str, default='fedgcdr')
parser.add_argument('--gnn_type', type=str, default='lightgcn', 
                    choices=['gat', 'lightgcn', 'graphsage', 'simgcl'],
                    help='选择使用的图神经网络模型: gat或lightgcn')
parser.add_argument('--knowledge', type=bool, default=False)
parser.add_argument('--only_ft', type=bool, default=False)
parser.add_argument('--eps', type=float, default=8)
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--num_users', type=int)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--description', type=str, default=None)
# Checkpoint相关参数
parser.add_argument('--resume_from', type=str, choices=['kg', 'kt', None], default=None,
                    help='从checkpoint恢复训练: kg=知识获取阶段后, kt=知识转移阶段后')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='checkpoint文件路径(用于resume_from)')
parser.add_argument('--save_checkpoint', action='store_true', default=True,
                    help='是否保存checkpoint(默认保存)')
parser.add_argument('--list_checkpoints', action='store_true',
                    help='列出所有可用的checkpoint')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='checkpoint保存目录')
args = parser.parse_args()

# 设置随机种子，确保可复现性
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 初始化Checkpoint管理器
checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir, max_keep=3)

# 列出checkpoint
if args.list_checkpoints:
    checkpoint_manager.print_checkpoints()
    exit(0)

# 检查恢复训练参数
if args.resume_from and not args.checkpoint_path:
    # 如果指定了resume_from但没有指定路径，列出可用的checkpoint
    print(f"错误: 使用--resume_from时必须指定--checkpoint_path")
    print(f"\n可用的checkpoint:")
    checkpoint_manager.print_checkpoints()
    exit(1)

# 根据gnn_type参数选择对应的模块
# 根据 gnn_type 参数选择对应的模型 - 使用模型注册表机制
# 根据 gnn_type 参数选择对应的模型 - 使用模型注册表机制

print(f'使用 {get_model_display_name(args.gnn_type)} 模型')
try:
    Server = get_server_class(args.gnn_type)
    Client = get_client_class(args.gnn_type)
    MLP = get_model_class(args.gnn_type + '_mlp')
except KeyError as e:
    print(f"错误：{e}")
    print(f"可用的模型：{list_all_models()}")
    exit(1)
device = torch.device(args.device)

domain_user, dic, domain_names = utility.set_dataset(args)
client_train_data, server_evaluate_data, num_items, num_users, user_dic = dic['client_train_data'], dic[
    'server_evaluate_data'], dic['num_items'], dic['num_users'], dic['user_dic']
clients = [Client(i, client_train_data[i], num_items, 0, domain_names, args) for i in range(args.num_users)]
server = [
    Server(i, domain_names[i], num_items[i], clients, domain_user[domain_names[i]], server_evaluate_data[i], user_dic,
           args) for i in range(args.num_domain)]
MLPs = [MLP(args.embedding_size).to(device) for _ in range(args.num_domain)]

# eval pre-train model
print(f'\n{"="*60}')
print(f'开始训练: 模型类型={args.gnn_type}, 目标域={domain_names[args.target_domain]}')
print(f'{"="*60}\n')
for it in server:
    it.test_mf(0)

tar_domain = args.target_domain
k_dic, emb_dic = {}, {}

# 根据checkpoint恢复情况设置跳过标志
skip_kg_training = args.resume_from in ['kg', 'kt']  # 跳过知识获取阶段
skip_kt_training = args.resume_from == 'kt'  # 跳过知识转移阶段

now = datetime.datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_').replace(':', "_")
formatted_date = now.strftime("%Y-%m-%d")
output_file = 'output/' + str(args.num_domain) + '_' + args.model + '_dp_' + str(args.dp) + '_tar_' + str(
    args.target_domain) + '_' + str(
    args.random_seed) + '_' + formatted_date_time + '.out'

# 获取git版本信息
git_commit_hash = get_git_commit_hash()

with open(output_file, 'w') as f:
    f.write(str(args) + f'\nGit Commit: {git_commit_hash}\n')
print(args)
print(f'Git Commit: {git_commit_hash}')

# load knowledge
if args.resume_from == 'kg':
    # 从知识获取阶段checkpoint恢复，跳过知识获取阶段
    print(f"\n{'='*60}")
    print(f"从知识获取阶段Checkpoint恢复训练")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"{'='*60}")

    # 加载checkpoint
    metadata, model_states, knowledge = checkpoint_manager.load_checkpoint(
        args.checkpoint_path, device
    )

    # 验证参数兼容性
    is_valid, message = checkpoint_manager.validate_checkpoint(metadata, args)
    if not is_valid:
        print(f"错误: {message}")
        exit(1)

    # 恢复模型状态
    restore_from_checkpoint(server, clients, model_states, knowledge, device, args)

    # 初始化目标域的MLP（知识转移阶段需要）
    server[tar_domain].mlp = MLPs

    print(f"✓ 已跳过知识获取阶段，直接进入知识转移阶段\n")

elif args.resume_from == 'kt':
    skip_kg_training = True
    skip_kt_training = True
    # 从知识转移阶段checkpoint恢复，跳过知识获取和转移阶段
    # 从知识转移阶段checkpoint恢复，跳过知识获取和转移阶段
    print(f"\n{'='*60}")
    print(f"从知识转移阶段Checkpoint恢复训练")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"{'='*60}")

    # 加载checkpoint
    metadata, model_states, knowledge, target_state, mlp_states = checkpoint_manager.load_checkpoint(
        args.checkpoint_path, device
    )

    # 验证参数兼容性
    is_valid, message = checkpoint_manager.validate_checkpoint(metadata, args)
    if not is_valid:
        print(f"错误: {message}")
        exit(1)

    # 恢复模型状态
    restore_from_checkpoint(server, clients, model_states, knowledge, device, args)

    # 恢复目标域状态
    server[tar_domain].mlp = MLPs
    restore_target_domain(server[tar_domain], MLPs, target_state, mlp_states, device)

    print(f"✓ 已跳过知识获取和转移阶段，直接进入微调阶段\n")

elif args.knowledge:
    skip_kg_training = False
    skip_kt_training = False
    with open('knowledge_hr/' + str(args.num_domain) + 'domains.json', 'r') as f:
        k_dic = json.load(f)
    for i in range(args.num_users):
        clients[i].knowledge = k_dic[str(i)]
else:
    skip_kg_training = False
    skip_kt_training = False
    order = [i for i in range(args.num_domain)]
    for it in order:
        max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
        knowledge = [1] * args.num_users
        for i in range(args.round_gat):
            model_name = get_model_display_name(args.gnn_type)
            print(f'{server[it].domain_name} {model_name} round {i}: ' + formatted_date_time)
            if args.gnn_type == 'lightgcn':
                server[it].kt_stage(round_id=i)
            else:
                server[it].kt_stage()
            hr_5, ndcg_5, hr_10, ndcg_10 = server[it].test_gnn(i)
            model_name = get_model_display_name(args.gnn_type)
            with open(output_file, 'a') as f:
                f.write(
                    f'[{server[it].domain_name} {model_name} Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                    f' ndcg_10 = {ndcg_10:.4f}\n')
            print(
                f'[{server[it].domain_name} {model_name} Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                f' ndcg_10 = {ndcg_10:.4f}\n')
            if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
                no_improve = 0
                epoch_id = i
                max_hr = hr_10
                max_ndcg = ndcg_10
                for client in clients:
                    knowledge[client.id] = client.knowledge[it]
            else:
                no_improve += 1
            # if no_improve > 100:
            #     break
        for client in clients:
            client.knowledge[it] = knowledge[client.id]

    # save_knowledge
    for i in range(args.num_users):
        for kl in clients[i].knowledge:
            if len(kl) != 0:
                kl[0] = kl[0].tolist()
        k_dic[i] = clients[i].knowledge
    with open('knowledge_64/' + str(args.num_domain) + 'domains' + '_' + formatted_date + '.json', 'w') as f:
        json.dump(k_dic, f)
    server[tar_domain].mlp = MLPs

    # 保存知识获取阶段checkpoint
    if args.save_checkpoint and not args.resume_from:
        kg_metrics = {
            'max_hr': float(max_hr) if isinstance(max_hr, (int, float)) else 0.0,
            'max_ndcg': float(max_ndcg) if isinstance(max_ndcg, (int, float)) else 0.0,
            'best_epoch': int(epoch_id) if isinstance(epoch_id, (int, float)) else 0,
        }
        checkpoint_path = checkpoint_manager.save_kg_checkpoint(server, clients, args, kg_metrics)
        print(f"✓ 知识获取阶段Checkpoint已保存\n")

# ASYNC(目标域知识激活)
if args.only_ft is False and not skip_kt_training:
    max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
    training_success = False
    for i in range(args.round_gat):
        model_name = get_model_display_name(args.gnn_type)
        print(f'{server[tar_domain].domain_name} {model_name} round {i}: ' + formatted_date_time)

        try:
            if args.gnn_type == 'lightgcn':
                server[tar_domain].kt_stage(True, i)
            else:
                server[tar_domain].kt_stage(True)
            hr_5, ndcg_5, hr_10, ndcg_10 = server[tar_domain].test_gnn(i)
            training_success = True
        except Exception as e:
            print(f'知识转移阶段发生错误: {str(e)}')
            import traceback
            traceback.print_exc()
            break
        with open(output_file, 'a') as f:
            f.write(
                f'[{server[tar_domain].domain_name} {model_name} Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
                f' ndcg_10 = {ndcg_10:.4f}\n')
        print(
            f'[{server[tar_domain].domain_name} {model_name} Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = {hr_10:.4f},'
            f' ndcg_10 = {ndcg_10:.4f}\n')
        if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
            no_improve = 0
            epoch_id = i
            max_hr = hr_10
            max_ndcg = ndcg_10
            emb_dic[domain_names[tar_domain]] = [server[tar_domain].user_embedding_with_attention.data.tolist(),
                                                 server[tar_domain].V.data.tolist()]
        else:
            no_improve += 1
        # if no_improve > 100:
        #     break

    # 保存最佳嵌入，如果没有则使用当前嵌入
    if domain_names[tar_domain] in emb_dic:
        server[tar_domain].U = torch.tensor(emb_dic[domain_names[tar_domain]][0], device=args.device)
        server[tar_domain].V = torch.tensor(emb_dic[domain_names[tar_domain]][1], device=args.device)
        print(f'成功加载{domain_names[tar_domain]}的嵌入')
    else:
        print(f'警告: 没有找到{domain_names[tar_domain]}的嵌入，使用当前嵌入')
        emb_dic[domain_names[tar_domain]] = [server[tar_domain].U.data.tolist(),
                                             server[tar_domain].V.data.tolist()]
    emb_dic['parser'] = vars(args)
    with open('embedding/' + args.model + '/' + str(args.num_domain) + 'dp' + str(args.dp) + '_' + args.dataset + '_' + domain_names[
        tar_domain] + '_' + args.model + '.json', 'w') as f:
        json.dump(emb_dic, f)

    # 保存知识转移阶段checkpoint
    if args.save_checkpoint and not args.resume_from:
        kt_metrics = {
            'max_hr': float(max_hr) if isinstance(max_hr, (int, float)) else 0.0,
            'max_ndcg': float(max_ndcg) if isinstance(max_ndcg, (int, float)) else 0.0,
            'best_epoch': int(epoch_id) if isinstance(epoch_id, (int, float)) else 0,
        }
        checkpoint_path = checkpoint_manager.save_kt_checkpoint(server, clients, tar_domain, MLPs, args, kt_metrics)
        print(f"✓ 知识转移阶段Checkpoint已保存\n")

# 加载目标域嵌入（仅在仅微调模式时，且不是从kt checkpoint恢复时）
# 注意：从kt checkpoint恢复时，嵌入已经从checkpoint中恢复，无需再从json加载
if args.only_ft and not args.resume_from:
    embedding_file = 'embedding/' + args.model + '/' + str(args.num_domain) + 'dp' + str(args.dp) + '_' + args.dataset + '_' + \
              domain_names[tar_domain] + '_' + args.model + '.json'
    print(f'从文件加载目标域嵌入: {embedding_file}')
    with open(embedding_file, 'r') as f:
        dic = json.load(f)
        tar_name = domain_names[args.target_domain]
        server[tar_domain].U.data, server[tar_domain].V.data = torch.tensor(dic[tar_name][0], device=args.device), \
            torch.tensor(dic[tar_name][1], device=args.device)
    print(f'✓ 目标域嵌入加载完成\n')

max_hr, max_ndcg, epoch_id, no_improve = 0, 0, 0, 0
max_hr_5, max_hr_10, max_ndcg_5, max_ndcg_10 = 0, 0, 0, 0
for i in range(args.round_ft):
    print(f'{server[tar_domain].domain_name} fine-tuning round {i} ' + formatted_date_time)
    server[tar_domain].mf_train()
    hr_5, ndcg_5, hr_10, ndcg_10 = server[tar_domain].test_mf(i)
    with open(output_file, 'a') as f:
        f.write(f'[{server[tar_domain].domain_name} Fine-tuning Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, '
                f'hr_10 ={hr_10:.4f}, ndcg_10 = {ndcg_10:.4f}\n')
    print(f'[{server[tar_domain].domain_name} Fine-tuning Round {i}] hr_5 = {hr_5:.4f}, ndcg_5 = {ndcg_5:.4f}, hr_10 = '
          f'{hr_10:.4f}, ndcg_10 = {ndcg_10:.4f}\n')
    max_hr_5 = max(max_hr_5, hr_5)
    max_hr_10 = max(max_hr_10, hr_10)
    max_ndcg_5 = max(max_ndcg_5, ndcg_5)
    max_ndcg_10 = max(max_ndcg_10, ndcg_10)
    if hr_10 > max_hr or (hr_10 == max_hr and ndcg_10 > max_ndcg):
        no_improve = 0
        epoch_id = i
        max_hr = hr_10
        max_ndcg = ndcg_10
    else:
        no_improve += 1

with open(output_file, 'a') as f:
    f.write(str(epoch_id) + '\n')
    f.write(f'hr_5 = {max_hr_5}, ndcg_5 = {max_ndcg_5}, hr_10 = {max_hr_10}, ndcg_10 = {max_ndcg_10}')
print(epoch_id)
print(f'hr_5 = {max_hr_5}, ndcg_5 = {max_ndcg_5}, hr_10 = {max_hr_10}, ndcg_10 = {max_ndcg_10}')
