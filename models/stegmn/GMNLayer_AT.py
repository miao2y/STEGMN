from re import X
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

class GMNLayer_AT(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,
                 learnable=False, with_mask=False):
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
        super(GMNLayer_AT, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.learnable = learnable
        self.with_mask = with_mask

        edge_coords_nf = 1

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

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        # Trans 是坐标的变化
        # 公式是：坐标差 * mlp(m_ij)
        # 即 4 个通道，每个通道都乘以一个数
        # coord_diff
        # trans 就是计算好的 m_ij * (X_j^l - X_i^l)
        # 再后面一步是求和，变成 x_i^l+1 = x_i^l + 1/N * sum(m_ij * (x_j^l - x_i^l))
        # trans 是 [11260,4,3]， 表示每个残基内 4 个原子，两两之间相减，再乘以 m_ij
        trans = coord_diff * self.coord_mlp(edge_feat) # * self.adj_mlp(edge_attr)#**
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
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))

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
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, dim=-1, keepdim=True)

        if self.norm_diff:
            coord_diff = F.normalize(coord_diff, p=2, dim=-1)
        # 径向距离、径向向量
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
    
        T = x.shape[0]
        x_list = []
        v_list = []
        for t in range(T):
            x_i = x[t]  # [N,  3]
            v_i = v[t]  # [N,  3]
            f_i = f[t]  # [N, 3]
            h_i = h[t]  # [N, n_hidden]
            # todo: 修改为多时间步的模式
            x_t_out, v_t_out = self.update_single_timestep(t,x, v, f, h,attention_weights, node_index, type)
            x_list.append(x_t_out)
            v_list.append(v_t_out)
        
        x_out = torch.stack(x_list, dim=0)  # [T, N, 4, 3]
        v_out = torch.stack(v_list, dim=0)  # [T, N, 4, 3]
        return x_out, v_out

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

        if type == 'Isolated':
            _x, _v, _f, _h = x[node_index], v[node_index], f[node_index], h[node_index]
            _a = _f / 1.
            _v = self.coord_mlp_vel(_h) * _v + _a
            _x = _x + _v
            # put the updated x, v (local object) back to x, v (global graph)
            x[node_index] = _x
            v[node_index] = _v
            return x, v

        elif type == 'Stick':
            id1, id2 = node_index[..., 0], node_index[..., 1]
            _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
            _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]
            _x0, _v0, _f0 = (_x1 + _x2) / 2, (_v1 + _v2) / 2, _f1 + _f2

            def apply_f(cur_x, cur_v, cur_f):
                _X = torch.stack((cur_f,), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.f_stick_mlp(_invariant_X)
                message = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                return message

            messages = [apply_f(_x1, _v1, _f1), apply_f(_x2, _v2, _f2)]
            _a0 = sum(messages) / len(messages)

            if self.learnable:
                _X = torch.stack((_a0, _x1 - _x0, _f1), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.stick_v_fk_mlp(_invariant_X)
                delta_v = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                _v1 = self.coord_mlp_vel(_h1) * _v1 + delta_v
                _x1 = _x1 + _v1
                _X = torch.stack((_a0, _x2 - _x0, _f2), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.stick_v_fk_mlp(_invariant_X)
                delta_v = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                _v2 = self.coord_mlp_vel(_h2) * _v2 + delta_v
                _x2 = _x2 + _v2

            else:
                # J:[500,4,1] ✅
                # 原论文是[500,1]
                J = torch.sum((_x1 - _x0) ** 2, dim=-1, keepdim=True) + torch.sum((_x2 - _x0) ** 2, dim=-1, keepdim=True)

                # _beta1, _beta2:[500,4,3] ✅
                _beta1, _beta2 = torch.cross((_x1 - _x0), _f1) / J, torch.cross((_x2 - _x0), _f2) / J
                # 970,4,3 ✅
                _beta = _beta1 + _beta2  # sum pooling over local object  # [B*N', 3]
                
                # beta 就是需要我们计算的角加速度，在这里添加时域的注意力
                # beta 的 shape 是 [970, 4, 3], 就是 970 个
                # 但是时域的shape 是 [2130,10,10], 总共有 2130 个节点，每个节点有 10 个时间步
                # 所以需要对每个链接，收集 10 个时间步的角加速度
                # 首先将 beta 扩充到 2130 个（根据 node_index 的 shape 是 [970, 2]），其他的就是默认为 0
                # 计算注意力的value
                # 最后只根据 node_index 取我们要的值
                
                # 实现时域注意力机制
                # 1. 首先将 beta 扩充到 2130 个节点
                N_total = x_T.shape[1]  # 总节点数 2130
                beta_expanded = torch.zeros(N_total, 3, device=_beta.device)  # [2130, 4, 3]
                
                # 根据 node_index 将 beta 放到正确的位置
                id1, id2 = node_index[..., 0], node_index[..., 1]
                # 对于Stick类型，角加速度应该同时影响两个连接的节点
                # 将角加速度放到两个节点的位置
                beta_expanded[id1] = _beta  # 将角加速度放到第一个节点位置
                beta_expanded[id2] = _beta  # 将角加速度也放到第二个节点位置
                
                # 2. 收集所有时间步的角加速度
                beta_all_timesteps = torch.zeros(N_total, x_T.shape[0], 3, device=_beta.device)  # [2130, T, 4, 3]
                
                # 遍历所有时间步，计算每个时间步的角加速度
                for t_step in range(x_T.shape[0]):
                    x_t = x_T[t_step]  # [2130, 4, 3]
                    v_t = v_T[t_step]  # [2130, 4, 3]
                    f_t = f_T[t_step]  # [2130, 4, 3]
                    h_t = h_T[t_step]  # [2130, hidden_nf]
                    
                    # 计算当前时间步的角加速度
                    _x1_t, _v1_t, _f1_t, _h1_t = x_t[id1], v_t[id1], f_t[id1], h_t[id1]
                    _x2_t, _v2_t, _f2_t, _h2_t = x_t[id2], v_t[id2], f_t[id2], h_t[id2]
                    _x0_t, _v0_t, _f0_t = (_x1_t + _x2_t) / 2, (_v1_t + _v2_t) / 2, _f1_t + _f2_t
                    
                    # 计算当前时间步的 beta
                    J_t = torch.sum((_x1_t - _x0_t) ** 2, dim=-1, keepdim=True) + torch.sum((_x2_t - _x0_t) ** 2, dim=-1, keepdim=True)
                    _beta1_t, _beta2_t = torch.cross((_x1_t - _x0_t), _f1_t) / J_t, torch.cross((_x2_t - _x0_t), _f2_t) / J_t
                    _beta_t = _beta1_t + _beta2_t  # [970, 4, 3]
                    
                    # 将当前时间步的角加速度放到两个节点的位置
                    beta_all_timesteps[id1, t_step] = _beta_t
                    beta_all_timesteps[id2, t_step] = _beta_t
                
                # 3. 计算注意力的value
                # 将角加速度转换为注意力机制的value
                beta_reshaped = beta_all_timesteps.view(N_total, x_T.shape[0], -1)  # [2130, T, 12] (4*3=12)
                
                # 使用注意力权重计算所有时间步的beta
                # attention_weights: [N, T, T]
                # 意思是每个残基都对应 10 个时间步的注意力权重
                if attention_weights is not None:
                    # 直接使用完整的注意力权重矩阵
                    # attention_weights: [2130, T, T]
                    # beta_reshaped: [2130, T, 12]
                    # 计算所有时间步的注意力加权beta
                    beta_attended_all = torch.bmm(attention_weights, beta_reshaped)  # [2130, T, 12]
                    
                    # 取出当前时间步t的beta
                    beta_attended_t = beta_attended_all[:, t, :]  # [2130, 12]
                    beta_attended_t = beta_attended_t.view(N_total, 3)  # [2130, 4, 3]
                    
                    # 只取我们需要的节点的角加速度
                    # 对于Stick类型，我们取两个节点的平均值作为最终的角加速度
                    _beta_attended_1 = beta_attended_t[id1]  # [970, 4, 3]
                    _beta_attended_2 = beta_attended_t[id2]  # [970, 4, 3]
                    
                    # 使用两个节点注意力加权的角加速度的平均值
                    _beta = _beta_attended_1 + _beta_attended_2


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
                _w = self.coord_mlp_w_vel(_h_c) * _w + _beta  # [B*N', 3]
                _v0 = self.coord_mlp_vel(_h_c) * _v0 + _a0  # [B*N', 3]
                _x0 = _x0 + _v0
                _theta = torch.norm(_w, p=2, dim=-1)  # [B*N']
                rot = self.compute_rotation_matrix(_theta, F.normalize(_w, p=2, dim=-1))

                _r = torch.einsum('bij,bjk->bik', rot, _r.unsqueeze(-1)).squeeze(-1)  # [B*N', 3]
                _x1 = _x0 + _r
                _x2 = _x0 - _r
                _v1 = _v0 + torch.cross(_w, _r)
                _v2 = _v0 + torch.cross(_w, - _r)

            # put the updated x, v (local object) back to x, v (global graph)
            x[id1], x[id2] = _x1, _x2
            v[id1], v[id2] = _v1, _v2
            return x, v

        elif type == 'Hinge':
            id0, id1, id2 = node_index[..., 0], node_index[..., 1], node_index[..., 2]
            _x0, _v0, _f0, _h0 = x[id0], v[id0], f[id0], h[id0]
            _x1, _v1, _f1, _h1 = x[id1], v[id1], f[id1], h[id1]
            _x2, _v2, _f2, _h2 = x[id2], v[id2], f[id2], h[id2]

            def apply_f(cur_x, cur_v, cur_f):
                _X = torch.stack((cur_f, cur_x - _x0, cur_v - _v0), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.f_hinge_mlp(_invariant_X)
                message = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                return message

            messages = [apply_f(_x0, _v0, _f0), apply_f(_x1, _v1, _f1), apply_f(_x2, _v2, _f2)]
            _a0 = sum(messages) / len(messages)

            def apply_g(cur_x, cur_f):
                message = torch.cross(cur_x - _x0, cur_f - _a0) / torch.sum((cur_x - _x0) ** 2, dim=-1, keepdim=True)
                return message

            _beta1, _beta2 = apply_g(_x1, _f1), apply_g(_x2, _f2)

            def compute_c_metrics(cur_x, cur_v):
                cur_r, relative_v = cur_x - _x0, cur_v - _v0
                cur_w = torch.cross(F.normalize(cur_r, dim=-1, p=2), relative_v) / torch.norm(
                    cur_r, dim=-1, p=2, keepdim=True).clamp_min(1e-5)
                return cur_r, cur_w

            _r1, _w1 = compute_c_metrics(_x1, _v1)
            _r2, _w2 = compute_c_metrics(_x2, _v2)

            trans_h1, trans_h2 = self.center_mlp(_h1), self.center_mlp(_h2)
            _h_c = trans_h1 + trans_h2
            _v0 = self.coord_mlp_vel(_h_c) * _v0 + _a0  # [B*N', 3]
            _x0 = _x0 + _v0

            if self.learnable:
                _X = torch.stack((_a0, _x1 - _x0, _f1), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.hinge_v_fk_mlp(_invariant_X)
                delta_v = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                _v1 = self.coord_mlp_vel(_h1) * _v1 + delta_v
                _x1 = _x1 + _v1
                _X = torch.stack((_a0, _x2 - _x0, _f2), dim=-1)
                _invariant_X = torch.einsum('bij,bjk->bik', _X.permute(0, 2, 1), _X)
                _invariant_X = _invariant_X.reshape(-1, _invariant_X.shape[-2] * _invariant_X.shape[-1])
                _invariant_X = F.normalize(_invariant_X, dim=-1, p=2)
                message = self.hinge_v_fk_mlp(_invariant_X)
                delta_v = torch.einsum('bij,bjk->bik', _X, message.unsqueeze(-1)).squeeze(-1)
                _v2 = self.coord_mlp_vel(_h2) * _v2 + delta_v
                _x2 = _x2 + _v2

            else:
                def update_c_metrics(rot_func, cur_w, cur_beta, cur_r, cur_h):
                    cur_w = self.coord_mlp_w_vel(cur_h) * cur_w + cur_beta  # [B*N', 3]
                    cur_theta = torch.norm(cur_w, p=2, dim=-1)  # [B*N']
                    cur_rot = rot_func(cur_theta, F.normalize(cur_w, p=2, dim=-1))
                    cur_r = torch.einsum('bcij,bck->bci', cur_rot, cur_r.unsqueeze(-1)).squeeze(-1)  # [B*N', C, 3]
                    return cur_r, cur_w

                _r1, _w1 = update_c_metrics(self.compute_rotation_matrix, _w1, _beta1, _r1, _h1)
                _r2, _w2 = update_c_metrics(self.compute_rotation_matrix, _w2, _beta2, _r2, _h2)

                _x1, _x2 = _x0 + _r1, _x0 + _r2
                _v1, _v2 = _v0 + torch.cross(_w1, _r1), _v0 + torch.cross(_w2, _r2)

            # put the updated x, v (local object) back to x, v (global graph)
            x[id0], x[id1], x[id2] = _x0, _x1, _x2
            v[id0], v[id1], v[id2] = _v0, _v1, _v2

            return x, v
        else:
            raise NotImplementedError('Unknown object type:', type)

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
        x, y, z = torch.unbind(d, dim=-1)
        cos, sin = torch.cos(theta), torch.sin(theta)
        ret = torch.stack((
            cos + (1 - cos) * x * x,
            (1 - cos) * x * y - sin * z,
            (1 - cos) * x * z + sin * y,
            (1 - cos) * x * y + sin * z,
            cos + (1 - cos) * y * y,
            (1 - cos) * y * z - sin * x,
            (1 - cos) * x * z - sin * y,
            (1 - cos) * y * z + sin * x,
            cos + (1 - cos) * z * z,
        ), dim=-1)

        return ret.reshape(-1, 3, 3)  # [B*N, 3, 3]


    def forward(self, h, edge_index, x, vec, cfg, edge_attr=None, node_attr=None):
        # h.shape: 1300,10,16
        # edge_index:array
        # x.shape: 1300,10,3
        # vec.shape: [1300,10,3]
        # cfg: {"Stick": tensor..., "Isolated": tensor...}

        T, N, hidden_nf = h.shape
        
        # 计算Query, Key, Value
        # Q: [T, N, hidden_nf]
        q = self.q_mlp(h)
        
        # K: [T, N, hidden_nf] 
        k = self.k_mlp(h)
        
        # 计算注意力得分
        # 将Q, K, V重塑为 [N, T, hidden_nf] 以便在时间维度上计算注意力
        q = q.transpose(0, 1)  # [N, T, hidden_nf]
        k = k.transpose(0, 1)  # [N, T, hidden_nf] 
        
        # 计算注意力得分: score = Q * K^T / sqrt(d_k)
        # [N, T, hidden_nf] * [N, hidden_nf, T] = [N, T, T]
        score = torch.matmul(q, k.transpose(-2, -1)) / (hidden_nf ** 0.5)
        
        # 如果使用mask，应用subsequent_mask避免信息泄露
        if self.with_mask:
            mask = subsequent_mask(T).to(score.device)
            score = score.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(score, dim=-1)  # [N, T, T]

        # 继续执行GMNLayerX的标准流程
        # 使用注意力处理后的特征 h_attended
        row, col = edge_index
        radial, coord_diff = self.coord2radial_X(edge_index, x)
        edge_feat = self.edge_model_X(h, edge_index, radial, edge_attr)
        coord_out, f = self.coord_model_X(x, edge_index, coord_diff, edge_feat)
        
        for type in cfg:
            x, vec = self.update(x, vec, f, h,attention_weights, node_index=cfg[type], type=type)
        h_out, agg = self.node_model_X(h, edge_index, edge_feat, node_attr, others=h)
        
        
        return h_out, x, vec
