import torch
import json
import os
import shutil
from datetime import datetime
import numpy as np


class CheckpointManager:
    """Checkpoint管理器，用于保存和加载训练状态"""

    def __init__(self, checkpoint_dir='checkpoints', max_keep=3):
        """
        初始化Checkpoint管理器

        Args:
            checkpoint_dir: checkpoint保存目录
            max_keep: 保留最近多少个checkpoint（默认3个）
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _generate_checkpoint_name(self, prefix, args):
        """生成checkpoint目录名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_info = f"{args.num_domain}domains"
        if hasattr(args, 'target_domain'):
            domain_info += f"_target{args.target_domain}"
        return f"{prefix}_{args.dataset}_{domain_info}_{timestamp}"

    def _cleanup_old_checkpoints(self, prefix):
        """清理旧的checkpoint，只保留最近的max_keep个"""
        pattern = f"{prefix}_"
        checkpoints = []

        for dir_name in os.listdir(self.checkpoint_dir):
            if dir_name.startswith(pattern) and os.path.isdir(os.path.join(self.checkpoint_dir, dir_name)):
                checkpoint_path = os.path.join(self.checkpoint_dir, dir_name)
                # 从目录名中提取时间戳
                try:
                    timestamp_str = dir_name.split('_')[-1]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    checkpoints.append((timestamp, checkpoint_path))
                except:
                    pass

        # 按时间排序，删除最旧的
        checkpoints.sort(reverse=True)
        for _, checkpoint_path in checkpoints[self.max_keep:]:
            print(f"删除旧checkpoint: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)

    def save_kg_checkpoint(self, servers, clients, args, metrics=None):
        """
        保存知识获取阶段checkpoint

        Args:
            servers: 服务器对象列表
            clients: 客户端对象列表
            args: 训练参数
            metrics: 可选的性能指标字典

        Returns:
            checkpoint_path: checkpoint保存路径
        """
        checkpoint_name = self._generate_checkpoint_name('kg', args)
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"保存知识获取阶段Checkpoint: {checkpoint_name}")
        print(f"{'='*60}")

        # 1. 保存元数据
        metadata = {
            'stage': 'knowledge_acquisition',
            'timestamp': datetime.now().isoformat(),
            'args': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                    for k, v in vars(args).items()},
        }
        if metrics:
            metadata['metrics'] = metrics

        with open(os.path.join(checkpoint_path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 2. 保存每个域的模型状态
        # 根据gnn_type获取模型属性名
        gnn_type = getattr(args, 'gnn_type', 'gat')
        if gnn_type in ['graphsage', 'simgcl']:
            item_model_attr = 'gnn_model'
        elif gnn_type == 'lightgcn':
            item_model_attr = 'item_lightgcn'
        else:  # gat
            item_model_attr = 'item_gat'

        model_states = {}
        for i, server in enumerate(servers):
            domain_key = f'domain_{i}_{server.domain_name}'
            model_states[domain_key] = {
                'U': server.U.cpu(),
                'V': server.V.cpu(),
                'item_model': getattr(server, item_model_attr).state_dict(),
                'item_model_attr': item_model_attr,  # 保存属性名以便恢复
                'user_embedding_with_attention': server.user_embedding_with_attention.cpu() if hasattr(server, 'user_embedding_with_attention') else None,
                'domain_attention': server.domain_attention.cpu() if hasattr(server, 'domain_attention') else None,
            }

        torch.save(model_states, os.path.join(checkpoint_path, 'models.pt'))
        print(f"✓ 保存模型状态: {len(model_states)} 个域")

        # 3. 保存客户端知识
        knowledge = {}
        for client in clients:
            client_key = f'client_{client.id}'
            # 将tensor转换为可序列化的格式
            client_knowledge = []
            for domain_knowledge in client.knowledge:
                if len(domain_knowledge) == 0:
                    client_knowledge.append([])
                else:
                    # knowledge[0]是tensor向量
                    knowledge_tensor = domain_knowledge[0]
                    if isinstance(knowledge_tensor, torch.Tensor):
                        client_knowledge.append([knowledge_tensor.cpu().numpy().tolist()])
                    else:
                        client_knowledge.append([knowledge_tensor])
            knowledge[client_key] = client_knowledge

        torch.save(knowledge, os.path.join(checkpoint_path, 'knowledge.pt'))
        print(f"✓ 保存客户端知识: {len(knowledge)} 个客户端")

        # 4. 保存训练状态摘要
        summary = {
            'total_domains': len(servers),
            'total_clients': len(clients),
            'domain_names': [server.domain_name for server in servers],
            'checkpoint_size_mb': self._get_dir_size(checkpoint_path),
        }
        with open(os.path.join(checkpoint_path, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✓ Checkpoint保存完成: {checkpoint_path}")
        print(f"  大小: {summary['checkpoint_size_mb']:.2f} MB")

        # 清理旧checkpoint
        self._cleanup_old_checkpoints('kg')

        return checkpoint_path

    def save_kt_checkpoint(self, servers, clients, target_domain_id, mlps, args, metrics=None):
        """
        保存知识转移阶段checkpoint

        Args:
            servers: 服务器对象列表
            clients: 客户端对象列表
            target_domain_id: 目标域ID
            mlps: MLP模型列表
            args: 训练参数
            metrics: 可选的性能指标字典

        Returns:
            checkpoint_path: checkpoint保存路径
        """
        checkpoint_name = self._generate_checkpoint_name('kt', args)
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"保存知识转移阶段Checkpoint: {checkpoint_name}")
        print(f"目标域: {servers[target_domain_id].domain_name}")
        print(f"{'='*60}")

        # 1. 保存元数据
        metadata = {
            'stage': 'knowledge_transfer',
            'target_domain_id': target_domain_id,
            'target_domain_name': servers[target_domain_id].domain_name,
            'timestamp': datetime.now().isoformat(),
            'args': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                    for k, v in vars(args).items()},
        }
        if metrics:
            metadata['metrics'] = metrics

        with open(os.path.join(checkpoint_path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 2. 保存所有域的模型状态（继承知识获取阶段）
        # 根据gnn_type获取模型属性名
        gnn_type = getattr(args, 'gnn_type', 'gat')
        if gnn_type in ['graphsage', 'simgcl']:
            item_model_attr = 'gnn_model'
        elif gnn_type == 'lightgcn':
            item_model_attr = 'item_lightgcn'
        else:  # gat
            item_model_attr = 'item_gat'

        model_states = {}
        for i, server in enumerate(servers):
            domain_key = f'domain_{i}_{server.domain_name}'
            model_states[domain_key] = {
                'U': server.U.cpu(),
                'V': server.V.cpu(),
                'item_model': getattr(server, item_model_attr).state_dict(),
                'item_model_attr': item_model_attr,
                'user_embedding_with_attention': server.user_embedding_with_attention.cpu(),
                'domain_attention': server.domain_attention.cpu(),
            }

        torch.save(model_states, os.path.join(checkpoint_path, 'models.pt'))
        print(f"✓ 保存模型状态: {len(model_states)} 个域")

        # 3. 保存客户端知识
        knowledge = {}
        for client in clients:
            client_key = f'client_{client.id}'
            client_knowledge = []
            for domain_knowledge in client.knowledge:
                if len(domain_knowledge) == 0:
                    client_knowledge.append([])
                else:
                    knowledge_tensor = domain_knowledge[0]
                    if isinstance(knowledge_tensor, torch.Tensor):
                        client_knowledge.append([knowledge_tensor.cpu().numpy().tolist()])
                    else:
                        client_knowledge.append([knowledge_tensor])
            knowledge[client_key] = client_knowledge

        torch.save(knowledge, os.path.join(checkpoint_path, 'knowledge.pt'))
        print(f"✓ 保存客户端知识: {len(knowledge)} 个客户端")

        # 4. 保存目标域的额外状态
        target_domain = servers[target_domain_id]
        target_state = {
            'user_embedding_with_attention': target_domain.user_embedding_with_attention.cpu(),
            'domain_attention': target_domain.domain_attention.cpu(),
        }
        torch.save(target_state, os.path.join(checkpoint_path, 'target_state.pt'))
        print(f"✓ 保存目标域状态")

        # 5. 保存MLP模型
        mlp_states = [mlp.state_dict() for mlp in mlps]
        torch.save(mlp_states, os.path.join(checkpoint_path, 'mlp.pt'))
        print(f"✓ 保存MLP模型: {len(mlp_states)} 个")

        # 6. 保存训练状态摘要
        summary = {
            'total_domains': len(servers),
            'total_clients': len(clients),
            'target_domain': target_domain.domain_name,
            'domain_names': [server.domain_name for server in servers],
            'checkpoint_size_mb': self._get_dir_size(checkpoint_path),
        }
        with open(os.path.join(checkpoint_path, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✓ Checkpoint保存完成: {checkpoint_path}")
        print(f"  大小: {summary['checkpoint_size_mb']:.2f} MB")

        # 清理旧checkpoint
        self._cleanup_old_checkpoints('kt')

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, device):
        """
        加载checkpoint

        Args:
            checkpoint_path: checkpoint路径
            device: 加载到的设备

        Returns:
            metadata: 元数据
            model_states: 模型状态
            knowledge: 客户端知识
            (可选) target_state: 目标域状态
            (可选) mlp_states: MLP状态
        """
        print(f"\n{'='*60}")
        print(f"加载Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        # 1. 加载元数据
        with open(os.path.join(checkpoint_path, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"✓ 阶段: {metadata['stage']}")
        print(f"✓ 时间: {metadata['timestamp']}")

        # 2. 加载模型状态
        model_states = torch.load(
            os.path.join(checkpoint_path, 'models.pt'),
            map_location=device
        )
        print(f"✓ 加载模型状态: {len(model_states)} 个域")

        # 3. 加载客户端知识
        knowledge = torch.load(
            os.path.join(checkpoint_path, 'knowledge.pt'),
            map_location=device
        )
        print(f"✓ 加载客户端知识: {len(knowledge)} 个客户端")

        result = [metadata, model_states, knowledge]

        # 4. 如果是知识转移阶段，加载额外数据
        if metadata['stage'] == 'knowledge_transfer':
            target_state = torch.load(
                os.path.join(checkpoint_path, 'target_state.pt'),
                map_location=device
            )
            result.append(target_state)
            print(f"✓ 加载目标域状态")

            mlp_states = torch.load(
                os.path.join(checkpoint_path, 'mlp.pt'),
                map_location=device
            )
            result.append(mlp_states)
            print(f"✓ 加载MLP模型: {len(mlp_states)} 个")

        print(f"✓ Checkpoint加载完成")

        return tuple(result)

    def validate_checkpoint(self, checkpoint_metadata, current_args):
        """
        验证checkpoint与当前参数的兼容性

        Args:
            checkpoint_metadata: checkpoint元数据
            current_args: 当前训练参数

        Returns:
            is_valid: 是否兼容
            message: 验证信息
        """
        # 检查关键字段
        critical_fields = ['dataset', 'num_domain', 'embedding_size', 'round_gat', 'round_ft']

        for field in critical_fields:
            checkpoint_value = checkpoint_metadata['args'].get(field)
            current_value = getattr(current_args, field, None)

            if checkpoint_value != current_value:
                return False, (
                    f"参数不匹配: {field}\n"
                    f"  Checkpoint: {checkpoint_value}\n"
                    f"  当前: {current_value}"
                )

        return True, "Checkpoint验证通过"

    def list_checkpoints(self):
        """列出所有可用的checkpoint"""
        checkpoints = {'kg': [], 'kt': []}

        for dir_name in os.listdir(self.checkpoint_dir):
            dir_path = os.path.join(self.checkpoint_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue

            # 检查元数据文件
            metadata_path = os.path.join(dir_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                continue

            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                stage = metadata.get('stage', 'unknown')
                timestamp = metadata.get('timestamp', '')
                size_mb = self._get_dir_size(dir_path)

                info = {
                    'name': dir_name,
                    'path': dir_path,
                    'stage': stage,
                    'timestamp': timestamp,
                    'size_mb': size_mb,
                }

                if stage == 'knowledge_acquisition':
                    checkpoints['kg'].append(info)
                elif stage == 'knowledge_transfer':
                    checkpoints['kt'].append(info)

            except Exception as e:
                print(f"警告: 无法读取checkpoint {dir_name}: {e}")

        # 按时间排序
        for stage in checkpoints:
            checkpoints[stage].sort(key=lambda x: x['timestamp'], reverse=True)

        return checkpoints

    def print_checkpoints(self):
        """打印所有可用的checkpoint"""
        checkpoints = self.list_checkpoints()

        print(f"\n{'='*80}")
        print(f"可用的Checkpoints (目录: {self.checkpoint_dir})")
        print(f"{'='*80}")

        for stage, stage_name in [('kg', '知识获取阶段'), ('kt', '知识转移阶段')]:
            print(f"\n【{stage_name}】")
            if not checkpoints[stage]:
                print("  无可用checkpoint")
            else:
                for i, cp in enumerate(checkpoints[stage], 1):
                    print(f"  {i}. {cp['name']}")
                    print(f"     路径: {cp['path']}")
                    print(f"     时间: {cp['timestamp']}")
                    print(f"     大小: {cp['size_mb']:.2f} MB")

        print(f"\n{'='*80}\n")

    def _get_dir_size(self, dir_path):
        """计算目录大小（MB）"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)


def restore_from_checkpoint(servers, clients, model_states, knowledge, device, args):
    """
    从checkpoint恢复模型状态

    Args:
        servers: 服务器对象列表
        clients: 客户端对象列表
        model_states: 模型状态字典
        knowledge: 知识字典
        device: 设备
        args: 训练参数
    """
    print(f"\n{'='*60}")
    print(f"恢复模型状态")
    print(f"{'='*60}")

    # 1. 恢复服务器模型状态
    for i, server in enumerate(servers):
        domain_key = f'domain_{i}_{server.domain_name}'
        if domain_key not in model_states:
            print(f"警告: 未找到域 {i} ({server.domain_name}) 的状态")
            continue

        state = model_states[domain_key]

        # 恢复嵌入
        server.U.data = state['U'].to(device)
        server.V.data = state['V'].to(device)

        # 恢复GNN模型（根据保存的item_model_attr动态选择）
        item_model_attr = state.get('item_model_attr', 'item_lightgcn')
        if hasattr(server, item_model_attr):
            getattr(server, item_model_attr).load_state_dict(state['item_model'])
        else:
            print(f"警告: 服务器没有属性 {item_model_attr}")

        # 恢复注意力相关状态（如果存在）
        if state.get('user_embedding_with_attention') is not None:
            server.user_embedding_with_attention.data = state['user_embedding_with_attention'].to(device)
        if state.get('domain_attention') is not None:
            server.domain_attention.data = state['domain_attention'].to(device)

        print(f"✓ 恢复域 {i} ({server.domain_name})")

    # 2. 恢复客户端知识
    for client in clients:
        client_key = f'client_{client.id}'
        if client_key not in knowledge:
            print(f"警告: 未找到客户端 {client.id} 的知识")
            continue

        client_knowledge_list = knowledge[client_key]
        for domain_id, domain_knowledge in enumerate(client_knowledge_list):
            if len(domain_knowledge) == 0:
                client.knowledge[domain_id] = []
            else:
                # 将list转换回tensor
                knowledge_array = np.array(domain_knowledge[0])
                knowledge_tensor = torch.from_numpy(knowledge_array).float().to(device)
                client.knowledge[domain_id] = [knowledge_tensor]

    print(f"✓ 恢复 {len(clients)} 个客户端的知识")

    print(f"✓ 模型状态恢复完成")


def restore_target_domain(server, mlps, target_state, mlp_states, device):
    """
    恢复目标域的额外状态

    Args:
        server: 目标域服务器对象
        mlps: MLP模型列表
        target_state: 目标域状态
        mlp_states: MLP状态列表
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"恢复目标域额外状态")
    print(f"{'='*60}")

    # 恢复目标域状态
    server.user_embedding_with_attention.data = target_state['user_embedding_with_attention'].to(device)
    server.domain_attention.data = target_state['domain_attention'].to(device)
    print(f"✓ 恢复目标域注意力状态")

    # 恢复MLP模型
    for i, (mlp, mlp_state) in enumerate(zip(mlps, mlp_states)):
        mlp.load_state_dict(mlp_state)
    print(f"✓ 恢复 {len(mlps)} 个MLP模型")

    print(f"✓ 目标域状态恢复完成")
