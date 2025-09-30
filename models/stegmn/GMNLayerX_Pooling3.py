from re import X
from torch import nn
import torch
import numpy as np
from models.gcl import unsorted_segment_mean_X, unsorted_segment_sum, unsorted_segment_mean
import torch.nn.functional as F


class GMNLayerX_Pooling3(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,
                 learnable=False, with_mask=False, num_future=1, num_past=10, num_channels=4,device=None):
        """
        The layer of Graph Mechanics Networks for multi-channel protein backbone.
        :param input_nf: input node feature dimension
        :param output_nf: output node feature dimension
        :param hidden_nf: hidden dimension
        :param edges_in_d: input edge dimension
        :param nodes_att_dim: attentional dimension, inherited
        :param act_fn: activation function
        :param recurrent: residual connection on x
        :param coords_weight: coords weight, inherited
        :param attention: use attention on edges, inherited
        :param norm_diff: normalize the distance, inherited
        :param tanh: Tanh activation, inherited
        :param learnable: use learnable FK
        :param num_channels: number of channels for protein backbone (default: 4)
        """
        super(GMNLayerX_Pooling3, self).__init__()
        
        self.num_channels = num_channels
        self.time_weights = nn.Parameter(torch.ones(10) / 10) 
        self.theta = nn.Parameter(torch.FloatTensor(num_future, num_past))
        self.to(device)
        self.reset_parameters()
    

    def reset_parameters(self):
        # time_weights init - 使用正常的初始化而不是设为0
        self.time_weights.data.uniform_(-0.1, 0.1)
        # theta init - 使用正常的初始化而不是设为0
        self.theta.data.uniform_(-0.1, 0.1)

    def forward(self, h, edge_index, x, vec, cfg, edge_attr=None, node_attr=None):
        # x, vec shape: [10, 1300, 4, 3] - 多通道输入，4个通道代表蛋白质主链的不同部分
        # cfg: 包含 'Isolated' 和 'Stick' 配置的字典
        # 获取最后一个时刻（t=10）的坐标
        x_last = x[-1]  # [1300, 4, 3] - 最后一个时刻的所有节点坐标
        vec_last = vec[-1]  # [1300, 4, 3] - 最后一个时刻的所有节点速度

        # x_hat 默认值，全部当作是 isolated 初始化默认值
        # 对多通道输入，我们需要对每个通道分别处理
        x_hat = torch.zeros_like(x_last)  # [1300, 4, 3]
        x_hat=torch.einsum("ij,jkts->ikts", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        print(" x_hat:")
        # -43 ~ 30
        print(x_hat.max(), x_hat.min())
        # for channel in range(self.num_channels):
        #     x_channel = x[:, :, channel, :]  # [10, 1300, 3] - 单个通道的时间序列
        #     x_last_channel = x_last[:, channel, :]  # [1300, 3] - 单个通道的最后时刻
            
        #     x_hat_channel = (
        #         torch.einsum("ij,jkt->ikt", self.theta, x_channel - x_last_channel.unsqueeze(0)).squeeze(0)
        #         + x_last_channel
        #     )
        #     # x_hat=torch.einsum("ij,jkts->ikts", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]

        #     x_hat[:, channel, :] = x_hat_channel

        # 对 isolated 的 x 保持不变
        if 'Isolated' in cfg:
            # isolated 节点保持不变
            pass
        
        # 对 stick 的 x，计算每个 x 相对 stick 中点的角速度
        if 'Stick' in cfg:
            stick_config = cfg['Stick']  # shape: [500, 2] 或其他形状
            
            if stick_config.shape[0] > 0:
                # 向量化处理所有stick对
                node1_indices = stick_config[:, 0].long()  # [num_sticks]
                node2_indices = stick_config[:, 1].long()  # [num_sticks]
                
                # 获取所有节点的位置和速度 [num_sticks, 10, 4, 3]
                x1_all = x[:, node1_indices, :, :]  # [10, num_sticks, 4, 3]
                x2_all = x[:, node2_indices, :, :]  # [10, num_sticks, 4, 3]
                v1_all = vec[:, node1_indices, :, :]  # [10, num_sticks, 4, 3]
                v2_all = vec[:, node2_indices, :, :]  # [10, num_sticks, 4, 3]
                
                # 计算所有stick的中点 [10, num_sticks, 4, 3]
                x_center_all = (x1_all + x2_all) / 2
                v_center_all = (v1_all + v2_all) / 2
                
                # 计算相对位置和相对速度 [10, num_sticks, 4, 3]
                r1_all = x1_all - x_center_all
                r2_all = x2_all - x_center_all
                v_rel1_all = v1_all - v_center_all
                v_rel2_all = v2_all - v_center_all
                
                # 对每个通道分别计算角速度
                # 计算角速度: ω = (r × v) / |r|² [10, num_sticks, 4, 3]
                # 使用更大的最小值来避免数值不稳定
                r1_norm = torch.norm(r1_all, dim=-1, keepdim=True).clamp_min(1e-3)
                r2_norm = torch.norm(r2_all, dim=-1, keepdim=True).clamp_min(1e-3)
                
                omega1_all = torch.cross(r1_all, v_rel1_all) / (r1_norm ** 2)
                omega2_all = torch.cross(r2_all, v_rel2_all) / (r2_norm ** 2)
                
                # 收集所有唯一的节点索引
                all_nodes = torch.cat([node1_indices, node2_indices]).unique()
                node_to_idx = {node.item(): idx for idx, node in enumerate(all_nodes)}
                
                # 初始化角速度存储张量 [num_unique_nodes, 10, 4, 3]
                angular_velocities = torch.zeros(len(all_nodes), 10, self.num_channels, 3, device=x.device)
                
                # 向量化角速度累积
                # 创建索引映射
                node1_positions = torch.tensor([node_to_idx[node.item()] for node in node1_indices], device=x.device, dtype=torch.long)
                node2_positions = torch.tensor([node_to_idx[node.item()] for node in node2_indices], device=x.device, dtype=torch.long)
                
                # 使用scatter_add_进行向量化累积
                angular_velocities.index_add_(0, node1_positions, omega1_all.permute(1, 0, 2, 3))  # [num_sticks, 10, 4, 3] -> [num_unique_nodes, 10, 4, 3]
                angular_velocities.index_add_(0, node2_positions, omega2_all.permute(1, 0, 2, 3))
                
                # 计算每个节点被包含在多少个stick中，用于平均
                node_counts = torch.zeros(len(all_nodes), device=x.device)
                node_counts.index_add_(0, node1_positions, torch.ones_like(node1_positions, dtype=torch.float))
                node_counts.index_add_(0, node2_positions, torch.ones_like(node2_positions, dtype=torch.float))
                
                # 平均角速度
                node_counts = node_counts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [num_unique_nodes, 1, 1, 1]
                angular_velocities = angular_velocities / node_counts.clamp_min(1)
                
                # 将角速度信息保存为类属性，以便后续使用
                self.angular_velocities = angular_velocities
                
                # 对每个节点的时间维度进行加权求和
                # 使用更稳定的权重初始化
                normalized_weights = torch.softmax(self.time_weights, dim=0)  # [10]
                angular_velocities_summed = torch.einsum('t,ntcd->ncd', normalized_weights, angular_velocities)  # [num_unique_nodes, 4, 3]
                
                # 向量化计算旋转矩阵 - 对每个通道分别计算
                def compute_rotation_matrices_vectorized_multi_channel(omegas, dt=1.0):
                    # omegas: [num_nodes, num_channels, 3]
                    omega_norms = torch.norm(omegas, dim=-1)  # [num_nodes, num_channels]
                    
                    # 创建单位矩阵 [num_nodes, num_channels, 3, 3]
                    batch_size, num_channels = omegas.shape[0], omegas.shape[1]
                    I = torch.eye(3, device=omegas.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_channels, -1, -1)
                    
                    # 处理角速度很小的节点
                    small_omega_mask = omega_norms < 1e-6
                    
                    # 归一化角速度方向
                    omega_unit = omegas / omega_norms.unsqueeze(-1).clamp_min(1e-6)  # [num_nodes, num_channels, 3]
                    
                    # 旋转角度
                    theta = omega_norms * dt  # [num_nodes, num_channels]
                    
                    # 使用Rodrigues公式计算旋转矩阵
                    cos_theta = torch.cos(theta)  # [num_nodes, num_channels]
                    sin_theta = torch.sin(theta)  # [num_nodes, num_channels]
                    
                    # 创建叉积矩阵 [num_nodes, num_channels, 3, 3]
                    omega_cross = torch.zeros(batch_size, num_channels, 3, 3, device=omegas.device)
                    omega_cross[:, :, 0, 1] = -omega_unit[:, :, 2]
                    omega_cross[:, :, 0, 2] = omega_unit[:, :, 1]
                    omega_cross[:, :, 1, 0] = omega_unit[:, :, 2]
                    omega_cross[:, :, 1, 2] = -omega_unit[:, :, 0]
                    omega_cross[:, :, 2, 0] = -omega_unit[:, :, 1]
                    omega_cross[:, :, 2, 1] = omega_unit[:, :, 0]
                    
                    # 计算K²
                    K_squared = torch.matmul(omega_cross, omega_cross)  # [num_nodes, num_channels, 3, 3]
                    
                    # 旋转矩阵 R = I + sin(θ)K + (1-cos(θ))K²
                    sin_theta_expanded = sin_theta.unsqueeze(-1).unsqueeze(-1)  # [num_nodes, num_channels, 1, 1]
                    cos_theta_expanded = cos_theta.unsqueeze(-1).unsqueeze(-1)  # [num_nodes, num_channels, 1, 1]
                    
                    R = I + sin_theta_expanded * omega_cross + (1 - cos_theta_expanded) * K_squared
                    
                    # 对于角速度很小的节点，使用单位矩阵
                    R[small_omega_mask] = I[small_omega_mask]
                    
                    return R
                
                # 批量计算旋转矩阵 [num_unique_nodes, 4, 3, 3]
                rotation_matrices = compute_rotation_matrices_vectorized_multi_channel(angular_velocities_summed, dt=1.0)
                
                # 计算所有stick的中心点 [num_sticks, 4, 3]
                x1_last_all = x_last[node1_indices]  # [num_sticks, 4, 3]
                x2_last_all = x_last[node2_indices]  # [num_sticks, 4, 3]
                x_center_last_all = (x1_last_all + x2_last_all) / 2  # [num_sticks, 4, 3]
                
                # 计算相对位置向量 [num_sticks, 4, 3]
                r1_last_all = x1_last_all - x_center_last_all
                r2_last_all = x2_last_all - x_center_last_all
                
                # 向量化计算新位置
                # 获取对应的旋转矩阵 [num_sticks, 4, 3, 3]
                R1_all = rotation_matrices[node1_positions]  # [num_sticks, 4, 3, 3]
                R2_all = rotation_matrices[node2_positions]  # [num_sticks, 4, 3, 3]
                
                # 批量计算旋转后的相对位置 [num_sticks, 4, 3]
                # 对每个通道分别进行矩阵乘法
                r1_rotated = torch.zeros_like(r1_last_all)
                r2_rotated = torch.zeros_like(r2_last_all)
                
                for channel in range(self.num_channels):
                    r1_rotated[:, channel, :] = torch.bmm(R1_all[:, channel, :, :], r1_last_all[:, channel, :].unsqueeze(-1)).squeeze(-1)
                    r2_rotated[:, channel, :] = torch.bmm(R2_all[:, channel, :, :], r2_last_all[:, channel, :].unsqueeze(-1)).squeeze(-1)
                
                # 计算新位置 [num_sticks, 4, 3]
                x1_next_all = x_center_last_all + r1_rotated
                x2_next_all = x_center_last_all + r2_rotated
                
                # 批量更新x_hat中对应节点的位置
                # 使用scatter操作批量更新
                all_node_indices = torch.cat([node1_indices, node2_indices])  # [2*num_sticks]
                all_next_positions = torch.cat([x1_next_all, x2_next_all])  # [2*num_sticks, 4, 3]
                # print("stick all_next_positions:")
                # print(all_next_positions.max(), all_next_positions.min())
                
                # 使用index_copy_来更新x_hat（如果有重复节点，取最后一次更新的值）
                x_hat.index_copy_(0, all_node_indices, all_next_positions)
                
                # 将x_hat保存为类属性，以便后续使用
                # self.x_hat = x_hat
                
        return x_hat
