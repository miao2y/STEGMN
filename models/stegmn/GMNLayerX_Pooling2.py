from torch import nn
import torch
import numpy as np
from models.gcl import unsorted_segment_mean_X, unsorted_segment_sum, unsorted_segment_mean
import torch.nn.functional as F


# subsequent_mask 是用于在 Transformer 中的 self-attention 机制中，避免当前位置的信息影响到未来位置的信息
# 返回一个上三角矩阵，对角线以下全是 1，对角线以上全是 0
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")

    return torch.from_numpy(subsequent_mask) == 0

class GMNLayerX_Pooling2(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,
                 learnable=False, with_mask=False,num_future=1,num_past=10):
        """
        The layer of Graph Mechanics Networks.
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
        """
        super(GMNLayerX_Pooling2, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.learnable = learnable
        self.with_mask = with_mask

        edge_coords_nf = 4 * 4

        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.k_mlp = nn.Sequential(
            nn.Linear(0 + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.v_mlp = nn.Sequential(
            nn.Linear(0 + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        n_basis_stick = 1
        n_basis_hinge = 3
        self.theta = nn.Parameter(torch.FloatTensor(num_future, num_past))
        # 初始化theta参数，使用较小的值避免数值过大
        self.theta.data.uniform_(-1, 1)
        # O init
        self.theta.data *= 0
        self.theta2 = nn.Parameter(torch.FloatTensor(num_future, num_past))
        # 初始化theta参数，使用较小的值避免数值过大
        self.theta2.data.uniform_(-1, 1)
        # O init
        self.theta2.data *= 0
        self.theta3 = nn.Parameter(torch.FloatTensor(num_future, num_past))
        # 初始化theta参数，使用较小的值避免数值过大
        self.theta3.data.uniform_(-1, 1)
        self.theta2.data *= 0

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.coord_mlp_w_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.center_mlp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, input_nf))

        self.f_stick_mlp = nn.Sequential(
            nn.Linear(n_basis_stick * n_basis_stick, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_stick)
        )
        self.f_hinge_mlp = nn.Sequential(
            nn.Linear(n_basis_hinge * n_basis_hinge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_basis_hinge)
        )

        if self.learnable:
            self.stick_v_fk_mlp = nn.Sequential(
                nn.Linear(3 * 3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 3)
            )
            self.hinge_v_fk_mlp = nn.Sequential(
                nn.Linear(3 * 3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 3)
            )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model_X(self, h, edge_index, radial, edge_attr):
        # h: torch.Size([10, 2130, 16])
        # edge_index: torch.Size([2, 11260])
        # radial: torch.Size([10, 11260, 16])
        # edge_attr: [10, 11260, edge_dim] 或 None
        
        row, col = edge_index
        
        # 遍历时间步，然后分别调用edge_model
        # 最后汇总起来成为一个tensor
        edge_feat_list = []
        for i in range(h.shape[0]):
            h_i = h[i]  # [2130, 16]
            radial_i = radial[i]  # [11260, 16]
            edge_attr_i = edge_attr[i] if edge_attr is not None else None  # [11260, edge_dim] 或 None
            
            # 获取当前时间步的source和target节点特征
            source_i = h_i[row]  # [11260, 16]
            target_i = h_i[col]  # [11260, 16]
            
            edge_feat_i = self.edge_model(source_i, target_i, radial_i, edge_attr_i)
            edge_feat_list.append(edge_feat_i)
        
        edge_feat = torch.stack(edge_feat_list, dim=0)  # [10, 11260, hidden_nf]
        return edge_feat

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, others=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        if others is not None:  # can concat h here
            agg = torch.cat([others, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def node_model_X(self, h, edge_index, edge_attr, node_attr, others=None):
        # h: torch.Size([10, 2130, hidden_nf])
        # edge_index: torch.Size([2, 11260])
        # edge_attr: torch.Size([10, 11260, hidden_nf])
        # node_attr: [10, 2130, node_attr_dim] 或 None
        # others: [10, 2130, hidden_nf] 或 None
        
        # 遍历时间步，然后分别调用node_model
        # 最后汇总起来成为一个tensor
        h_out_list = []
        agg_list = []
        for i in range(h.shape[0]):
            h_i = h[i]  # [2130, hidden_nf]
            edge_attr_i = edge_attr[i]  # [11260, hidden_nf]
            node_attr_i = node_attr[i] if node_attr is not None else None  # [2130, node_attr_dim] 或 None
            others_i = others[i] if others is not None else None  # [2130, hidden_nf] 或 None
            
            h_i_out, agg_i = self.node_model(h_i, edge_index, edge_attr_i, node_attr_i, others_i)
            h_out_list.append(h_i_out)
            agg_list.append(agg_i)
        
        h_out = torch.stack(h_out_list, dim=0)  # [10, 2130, output_nf]
        agg_out = torch.stack(agg_list, dim=0)  # [10, 2130, agg_dim]
        return h_out, agg_out
    
    def merge_pooled_results(self, x_isolated_pooled, vec_isolated_pooled, x_stick_pooled, vec_stick_pooled, 
                            isolated_indices, stick_indices, total_nodes, device):
        """
        将isolate_pooling和stick_pooling的结果合并
        :param x_isolated_pooled: [K_isolated, 4, 3] - Isolated节点的pooled坐标
        :param vec_isolated_pooled: [K_isolated, 4, 3] - Isolated节点的pooled速度
        :param x_stick_pooled: [N, 4, 3] - Stick节点的完整pooled坐标 (包含所有节点)
        :param vec_stick_pooled: [N, 4, 3] - Stick节点的完整pooled速度 (包含所有节点)
        :param isolated_indices: [K_isolated,] - Isolated节点的索引
        :param stick_indices: [K_stick, 2] - Stick节点的索引对
        :param total_nodes: int - 总节点数
        :param device: torch.device - 设备
        :return: x_final, vec_final - 合并后的完整结果
        """
        # stick_pooling返回的是完整结果，直接使用
        x_final = x_stick_pooled.clone()
        vec_final = vec_stick_pooled.clone()
        
        # 用isolate_pooling的结果替换对应的isolated节点
        if isolated_indices is not None and len(isolated_indices) > 0:
            x_final[isolated_indices] = x_isolated_pooled
            vec_final[isolated_indices] = vec_isolated_pooled
        
        return x_final, vec_final
    def isolate_pooling(self, x, v, f, h, node_index):
        """
        对Isolated节点应用时域pooling操作，将时间维度压缩为1个值
        :param x: position [T, N, 4, 3] - 整个时间序列的坐标
        :param v: velocity [T, N, 4, 3] - 整个时间序列的速度  
        :param f: force [T, N, 4, 3] - 整个时间序列的力
        :param h: node feature [T, N, hidden_nf] - 整个时间序列的特征
        :param node_index: [K,] - Isolated节点的索引
        :return: 更新后的x和v，只有Isolated节点被修改，时间维度压缩为1
        """
        if node_index is None or len(node_index) == 0:
            return x, v
        
        # 获取Isolated节点的索引
        isolated_indices = node_index
        
        # 提取Isolated节点的坐标
        x_isolated = x[:, isolated_indices]  # [T, K, 4, 3]
        x_diff_weighted=torch.einsum("ij,jkts->ikts", self.theta,x_isolated-x_isolated[-1].unsqueeze(0)).squeeze(0)+x_isolated[-1]
        
        # # 应用时域pooling操作，参考公式：x_hat=torch.einsum("ij,jkts->ikts", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        
        # # 计算相对于最后一个时间步的差值
        # x_diff = x_isolated - x_isolated[-1].unsqueeze(0)  # [T, K, 4, 3]
        
        # # 应用theta权重进行时域pooling，将时间维度从T压缩为1
        # # self.theta: [num_future, num_past] = [1, T]
        # # x_diff: [T, K, 4, 3]
        # # 使用einsum: "ij,jkts->ikts" 其中 i=1, j=T, k=K, t=4, s=3
        
        # # 对theta进行归一化，确保数值稳定性
        # theta_normalized = F.softmax(self.theta, dim=-1)  # 确保权重和为1
        # x_diff_weighted = torch.einsum("ij,jkts->ikts", theta_normalized, x_diff)  # [1, K, 4, 3]
        
        # 加上最后一个时间步的值，得到pooling后的结果
        # x_isolated_pooled = x_diff_weighted.squeeze(0) + x_isolated[-1]  # [K, 4, 3]
        
        # 直接返回pooling后的结果
        return x_diff_weighted, v[:, isolated_indices][-1]

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        # Trans 是坐标的变化
        # 公式是：坐标差 * mlp(m_ij)
        # 即 4 个通道，每个通道都乘以一个数
        # coord_diff
        # trans 就是计算好的 m_ij * (X_j^l - X_i^l)
        # 再后面一步是求和，变成 x_i^l+1 = x_i^l + 1/N * sum(m_ij * (x_j^l - x_i^l))
        # trans 是 [11260,4,3]， 表示每个残基内 4 个原子，两两之间相减，再乘以 m_ij
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(
            -1
        )  # * self.adj_mlp(edge_attr)#**
        # trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        # unsorted_segment_mean_X 是自定义的函数，用于计
        # num_segments 是节点的数量
        # row 是 [11260]
        # num_segments 是 2130，即残基的数量
        # unsorted_segment_mean_X(trans, row, 2130) 就是"统计每个节点收到的所有边的'力'，并取平均，得到每个节点的平均'力'"，用于后续节点坐标的更新。
        # 'segment'就是组，即每个原子
        # x_i^l+1 = x_i^l + 1/N * sum(m_ij * (x_j^l - x_i^l))
        # 有的公式没有1/N， 因为有的公式是 1/N * sum(m_ij * (x_j^l - x_i^l))
        # 这种情况下用 unsorted_segment_sum_X
        # agg: [2130, 4, 3]
        agg = unsorted_segment_mean_X(trans, row, num_segments=coord.size(0))

        f = agg * self.coords_weight
        coord_ = coord + f
        return coord_, f

    def coord_model_X(self, coord, edge_index, coord_diff, edge_feat):
        # coord: torch.Size([10, 2130, 4, 3])
        # edge_index: torch.Size([2, 11260])
        # coord_diff: torch.Size([10, 11260, 4, 3])
        # edge_feat: torch.Size([10, 11260, hidden_nf])
        
        # 遍历时间步，然后分别调用coord_model
        # 最后汇总起来成为一个tensor
        coord_list = []
        f_list = []
        for i in range(coord.shape[0]):
            coord_i = coord[i]  # [2130, 4, 3]
            coord_diff_i = coord_diff[i]  # [11260, 4, 3]
            edge_feat_i = edge_feat[i]  # [11260, hidden_nf]
            
            coord_i_out, f_i = self.coord_model(coord_i, edge_index, coord_diff_i, edge_feat_i)
            coord_list.append(coord_i_out)
            f_list.append(f_i)
        
        coord_out = torch.stack(coord_list, dim=0)  # [10, 2130, 4, 3]
        f_out = torch.stack(f_list, dim=0)  # [10, 2130, 4, 3]
        return coord_out, f_out

    def coord2radial(self, edge_index, coord):
        # batch_size 是 10，一张图有 1126 条边，因此总共是 11260 条边
        # row, col 是 [11260]， 表示每条边的两个端点
        row, col = edge_index
        # coord 是[ 11260, 4, 3], 表示径向向量，每个残基有 4 个原子
        # coord_diff 是[ 11260, 4, 3], 表示每个残基中对应原子之间相减，即 a1-a2, b1-b2, c1-c2, d1-d2
        # 也即径向向量
        coord_diff = coord[row] - coord[col]
        # coord_diff_pro 是[ 11260, 16], 表示每个残基中对应原子之间相减的平方和
        # 一条边对应了两边残基内 4 个原子之间的关系，每个原子都计算对面残基内 4 个原子之间的距离，总共有 4*4=16个距离
        # 因此 coord_diff_pro 是[ 11260, 16]
        coord_diff_pro = torch.einsum("ijt,ikt->ijk", coord_diff, coord_diff).reshape(
            coord_diff.shape[0], -1
        )
        # radial 是[ 11260, 16], 表示每个残基中对应原子之间相减的平方和的归一化
        radial = F.normalize(coord_diff_pro, dim=-1, p=2)

        return radial, coord_diff
    
    def coord2radial_X(self, edge_index, coord):
        # coord 是[10,2130,4,3]
        # edge_index 是 list[[11260], [11260]]
        row, col = edge_index
        
        # 遍历coord第一个维度，即时间步，然后分别调用coord2radial
        # 最后汇总起来成为一个tensor
        radial = []
        coord_diff = []
        for i in range(coord.shape[0]):
            radial_i, coord_diff_i = self.coord2radial(edge_index, coord[i])
            radial.append(radial_i)
            coord_diff.append(coord_diff_i)
        radial = torch.stack(radial, dim=0)
        coord_diff = torch.stack(coord_diff, dim=0)
        return radial, coord_diff

    def calculate_a(self, x, v, f, h, node_index, type='Isolated'):
        id1, id2 = node_index[..., 0], node_index[..., 1]
        _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
        _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]
        _x0, _v0, _f0 = (_x1 + _x2) / 2, (_v1 + _v2) / 2, _f1 + _f2

        def apply_f(cur_x, cur_v, cur_f):
            # cur_x, cur_v, cur_f 都是:[ 970, 4, 3]
            # _X：[970,4,3,1]
            _X = torch.stack((cur_f,), dim=-1)

            # _invariant_X 得是 [970,4,1]
            # _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
            _invariant_X = torch.norm(_X, dim=2, keepdim=True)  # [970, 4, 1, 1]
            # _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
            _invariant_X = _invariant_X.reshape(_invariant_X.shape[0], _invariant_X.shape[1], _invariant_X.shape[2] * _invariant_X.shape[3])
            _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
            message = self.f_stick_mlp(_invariant_X)
            # message = torch.einsum('bij,bjk->bik', _X.squeeze(-1), message.unsqueeze(-1)).squeeze(-1)
            message = torch.einsum('bcij,bcjk->bcik', _X, message.unsqueeze(-1)).squeeze(-1)
            return message

        messages = [apply_f(_x1, _v1, _f1), apply_f(_x2, _v2, _f2)]
        _a0 = sum(messages) / len(messages)

        # J:[500,4,1] ✅
        # 原论文是[500,1]
        J = torch.sum((_x1 - _x0) ** 2, dim=-1, keepdim=True) + torch.sum((_x2 - _x0) ** 2, dim=-1, keepdim=True)

        # _beta1, _beta2:[500,4,3] ✅
        _beta1, _beta2 = torch.cross((_x1 - _x0), _f1) / J, torch.cross((_x2 - _x0), _f2) / J
        # 970,4,3 ✅
        _beta = _beta1 + _beta2  # sum pooling over local object  # [B*N', 3]
        return _beta, id1, id2

    def update(self, x, v, f, h,attention_weights, node_index, type='Isolated'):
        """
        Update X and V given the current X, V, and force F
        :param x: position  [T, N, 4, 3] or [N, 4, 3] - 支持多时间步或单时间步
        :param v: velocity  [T, N, 4, 3] or [N, 4, 3] - 支持多时间步或单时间步
        :param f: force  [T, N, 4, 3] or [N, 4, 3] - 支持多时间步或单时间步
        :param h:  node feature  [T, N, n_hidden] or [N, n_hidden] - 支持多时间步或单时间步
        :param node_index: [K, 2] for stick, [K, 3] for hinge, [K,] for isolated, K is the number of rigid objects
        :param type:  the type of rigid objects, 'Isolated' or 'Stick' or 'Hinge'
        :return: the updated x and v with same shape as input
        """
        # 多时间步输入
        T = x.shape[0]
        x_t_out, v_t_out = self.update_single_timestep(T-1, x, v, f, h,attention_weights, node_index, type)
        return x_t_out, v_t_out
    


    def update_single_timestep(self, t, x_T, v_T, f_T, h_T,attention_weights, node_index, type='Isolated'):
        """
        Update X and V for a single timestep
        :param x: position  [N, 4, 3]
        :param v: velocity  [N, 4, 3]
        :param f: force  [N, 4, 3]
        :param h:  node feature  [N, n_hidden]
        :param node_index: [K, 2] for stick, [K, 3] for hinge, [K,] for isolated, K is the number of rigid objects
        :param type:  the type of rigid objects, 'Isolated' or 'Stick' or 'Hinge'
        :return: the updated x [N, 4, 3] and v [N, 4, 3]
        """
        x = x_T[t]
        v = v_T[t]
        f = f_T[t]
        h = h_T[t]

        id1, id2 = node_index[..., 0], node_index[..., 1]
        _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
        _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]
        _x0, _v0, _f0 = (_x1 + _x2) / 2, (_v1 + _v2) / 2, _f1 + _f2

        def apply_f(cur_x, cur_v, cur_f):
            # cur_x, cur_v, cur_f 都是:[ 970, 4, 3]
            # _X：[970,4,3,1]
            _X = torch.stack((cur_f,), dim=-1)

            # _invariant_X: [970,4,1]
            # _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
            _invariant_X = torch.norm(_X, dim=2, keepdim=True)  # [970, 4, 1, 1]
            # _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
            _invariant_X = _invariant_X.reshape(_invariant_X.shape[0], _invariant_X.shape[1], _invariant_X.shape[2] * _invariant_X.shape[3])
            _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
            message = self.f_stick_mlp(_invariant_X)
            # message = torch.einsum('bij,bjk->bik', _X.squeeze(-1), message.unsqueeze(-1)).squeeze(-1)
            message = torch.einsum('bcij,bcjk->bcik', _X, message.unsqueeze(-1)).squeeze(-1)
            return message

        messages = [apply_f(_x1, _v1, _f1), apply_f(_x2, _v2, _f2)]
        _a0 = sum(messages) / len(messages)



        # compute c metrics
        # 970,4,3 ✅
        _r, _v = (_x1 - _x2) / 2, (_v1 - _v2) / 2
        # 970,4,3 ✅
        _w = torch.cross(F.normalize(_r, dim=-1, p=2), _v) / torch.norm(_r, dim=-1, p=2, keepdim=True).clamp_min(
            1e-5)  # [B*N', 3]
        # 970,16 ✅
        trans_h1, trans_h2 = self.center_mlp(_h1), self.center_mlp(_h2)
        # 970,16
        _h_c = trans_h1 + trans_h2

        # 
        # _w = self.coord_mlp_w_vel(_h_c).unsqueeze(-1) * _w + _beta  # [B*N', 3]

        def calculate_w():
            # 计算所有时间的 _w
            # 遍历所有时间步，计算每个时间步的角速度
            w_all_timesteps = torch.zeros(x_T.shape[0], len(node_index), 4, 3, device=_w.device)  # [T, 970, 4, 3]
            
            for t_step in range(x_T.shape[0]):
                x_t = x_T[t_step]  # [2130, 4, 3]
                v_t = v_T[t_step]  # [2130, 4, 3]
                
                # 获取当前时间步的节点数据
                _x1_t, _v1_t = x_t[id1], v_t[id1]  # [970, 4, 3]
                _x2_t, _v2_t = x_t[id2], v_t[id2]  # [970, 4, 3]
                
                # 计算当前时间步的 _r 和 _v
                _r_t, _v_t = (_x1_t - _x2_t) / 2, (_v1_t - _v2_t) / 2
                
                # 计算当前时间步的角速度
                _w_t = torch.cross(F.normalize(_r_t, dim=-1, p=2), _v_t) / torch.norm(_r_t, dim=-1, p=2, keepdim=True).clamp_min(1e-5)
                
                # 存储当前时间步的角速度
                w_all_timesteps[t_step] = _w_t
            
            return w_all_timesteps

        w_all_timesteps = calculate_w()  # [T, 970, 4, 3]
        # 计算所有时间下对T的差值, 如果是 T 就是 0
        w_all_timesteps_diff = w_all_timesteps - w_all_timesteps[-1].unsqueeze(0)
        _w = torch.einsum("ij,jkts->ikts", self.theta3, w_all_timesteps_diff) + w_all_timesteps[-1].unsqueeze(0)
        _w = _w.squeeze(0)
        # print(_w.shape)
        # exit()
        
        # 取最后一个时间步的角速度作为当前时间步的角速度
        # _w = w_all_timesteps[-1]  # [970, 4, 3]

        
        _v0 = self.coord_mlp_vel(_h_c).unsqueeze(-1) * _v0 + _a0  # [B*N', 3]
        _x0 = _x0 + _v0
        _theta = torch.norm(_w, p=2, dim=-1)  # [B*N']
        # rot.shape：torch.Size([9700, 4, 3, 3])
        rot = self.compute_rotation_matrix(_theta, F.normalize(_w, p=2, dim=-1))
        # _r.unsqueeze(-1).shape： torch.Size([9700, 4, 3, 1])
        _r = torch.einsum('bcij,bcjk->bcik', rot, _r.unsqueeze(-1)).squeeze(-1)  # [B*N', C, 3]            
        _x1 = _x0 + _r
        _x2 = _x0 - _r
        _v1 = _v0 + torch.cross(_w, _r)
        _v2 = _v0 + torch.cross(_w, - _r)


        # put the updated x, v (local object) back to x, v (global graph)
        x[id1], x[id2] = _x1, _x2
        v[id1], v[id2] = _v1, _v2
        return x, v


    def update_X(self, x, v, f, h,attention_weights, node_index, type='Isolated'):
        # x: torch.Size([10, 2130, 4, 3]) - 整个时间序列的坐标
        # v: torch.Size([10, 2130, 4, 3]) - 整个时间序列的速度
        # f: torch.Size([10, 2130, 4, 3]) - 整个时间序列的力
        # h: torch.Size([10, 2130, hidden_nf]) - 整个时间序列的特征
        # node_index: [K, 2] for stick, [K, 3] for hinge, [K,] for isolated
        
        # 直接传递整个时间序列给update函数
        x_out, v_out = self.update(x, v, f, h,attention_weights, node_index, type)
        return x_out, v_out

    @staticmethod
    def compute_rotation_matrix(theta, d):
        # theta: [B, C] -> [B, C, 1] for broadcasting
        # d: [B, C, 3]
        theta = theta.unsqueeze(-1)  # [B, C, 1]
        
        # Unbind along the last dimension (3D coordinates)
        x, y, z = torch.unbind(d, dim=-1)  # Each is [B, C]
        
        cos, sin = torch.cos(theta), torch.sin(theta)  # [B, C, 1]
        
        # Compute rotation matrix elements for each channel
        ret = torch.stack((
            cos + (1 - cos) * x.unsqueeze(-1) * x.unsqueeze(-1),
            (1 - cos) * x.unsqueeze(-1) * y.unsqueeze(-1) - sin * z.unsqueeze(-1),
            (1 - cos) * x.unsqueeze(-1) * z.unsqueeze(-1) + sin * y.unsqueeze(-1),
            (1 - cos) * x.unsqueeze(-1) * y.unsqueeze(-1) + sin * z.unsqueeze(-1),
            cos + (1 - cos) * y.unsqueeze(-1) * y.unsqueeze(-1),
            (1 - cos) * y.unsqueeze(-1) * z.unsqueeze(-1) - sin * x.unsqueeze(-1),
            (1 - cos) * x.unsqueeze(-1) * z.unsqueeze(-1) - sin * y.unsqueeze(-1),
            (1 - cos) * y.unsqueeze(-1) * z.unsqueeze(-1) + sin * x.unsqueeze(-1),
            cos + (1 - cos) * z.unsqueeze(-1) * z.unsqueeze(-1),
        ), dim=-1)  # [B, C, 9]

        # Reshape to [B, C, 3, 3]
        return ret.reshape(*theta.shape[:-1], 3, 3)

    def forward(self, h, edge_index, x, vec, cfg, edge_attr=None, node_attr=None):
        # GMNLayerX_AT类似于GMNLayerX，但是区别在于输入会比GMNLayerX多一个维度 T,即时间序列长度 T
        # 计算注意力得分，在时间步 T 上计算，即每个残基都和 T 个残基计算注意力得分
        # 即对每个残基，收集 T 个时间步的角加速度，得到 T 个角加速度
        
        # 假设输入 h 的形状是 [T, N, hidden_nf]，其中 T 是时间序列长度，N 是节点数量
        # 如果输入不是这种形状，需要先调整
        if len(h.shape) == 2:
            # 如果输入是 [N, hidden_nf]，假设只有一个时间步
            h = h.unsqueeze(0)  # [1, N, hidden_nf]
        
        T, N, hidden_nf = h.shape
        
        # 计算Query, Key, Value
        # Q: [T, N, hidden_nf]
        q = self.q_mlp(h)
        
        # K: [T, N, hidden_nf] 
        k = self.k_mlp(h)
        
        # V: [T, N, hidden_nf]
        v = self.v_mlp(h)
        
        # 计算注意力得分
        # 将Q, K, V重塑为 [N, T, hidden_nf] 以便在时间维度上计算注意力
        q = q.transpose(0, 1)  # [N, T, hidden_nf]
        k = k.transpose(0, 1)  # [N, T, hidden_nf] 
        v = v.transpose(0, 1)  # [N, T, hidden_nf]
        
        # 计算注意力得分: score = Q * K^T / sqrt(d_k)
        # [N, T, hidden_nf] * [N, hidden_nf, T] = [N, T, T]
        score = torch.matmul(q, k.transpose(-2, -1)) / (hidden_nf ** 0.5)
        
        # 如果使用mask，应用subsequent_mask避免信息泄露
        if self.with_mask:
            mask = subsequent_mask(T).to(score.device)
            score = score.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(score, dim=-1)  # [N, T, T]

        # 继续执行GMNLayerX的流程
        row, col = edge_index
        radial, coord_diff = self.coord2radial_X(edge_index, x)
        edge_feat = self.edge_model_X(h, edge_index, radial, edge_attr)
        coord_out, f = self.coord_model_X(x, edge_index, coord_diff, edge_feat)
        
        
        x_stick, vec_stick = self.update_X(x, vec, f, h, attention_weights, node_index=cfg["Stick"], type="Stick")

        # 只取最后一个时刻的
        x_hat_isolated, vec_hat_isolated = self.isolate_pooling(x, vec, f, h, node_index=cfg['Isolated'])
        x_hat_stick = x_stick
        vec_hat_stick = vec_stick
        x_hat, vec = self.merge_pooled_results(x_hat_isolated, vec_hat_isolated, x_hat_stick, vec_hat_stick, cfg['Isolated'], cfg['Stick'], N,device=x.device)

        return x_hat
