from torch import nn
import torch
from models.gcl import unsorted_segment_mean_X, unsorted_segment_sum, unsorted_segment_mean
import torch.nn.functional as F

class GMNLayerX(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,
                 learnable=False):
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
        super(GMNLayerX, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.learnable = learnable
        edge_coords_nf = 4 * 4

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

    def update(self, x, v, f, h, node_index, type='Isolated'):
        """
        Update X and V given the current X, V, and force F
        :param x: position  [N, 3]
        :param v: velocity  [N, 3]
        :param f: force  [N, 3]
        :param h:  node feature  [N, n_hidden]
        :param node_index: [K, 2] for stick, [K, 3] for hinge, [K,] for isolated, K is the number of rigid objects
        :param type:  the type of rigid objects, 'Isolated' or 'Stick' or 'Hinge'
        :return: the updated x [N, 3] and v [N, 3]
        """
        if type == 'Isolated':
            _x, _v, _f, _h = x[node_index], v[node_index], f[node_index], h[node_index]
            _a = _f / 1.
            _v = self.coord_mlp_vel(_h).unsqueeze(-1) * _v + _a
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
                _w = self.coord_mlp_w_vel(_h_c).unsqueeze(-1) * _w + _beta  # [B*N', 3]
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

    def forward(self, h, edge_index, x, v, cfg, edge_attr=None, node_attr=None):
        """
        :param h: the node aggregated feature  [N, n_hidden]
        :param edge_index:  [2, M], M is the number of edges
        :param x: input coordinate  [N, 3]
        :param v: input velocity  [N, 3]
        :param cfg: {'isolated': idx, 'stick': [(c0, c1) ...] (K, 2), 'hinge': [(c0, c1, c2) ...] (K, 3)}. K is the number of rigid obj
        :param edge_attr: edge feature  [M, n_edge]
        :param node_attr: the node input feature  [N, n_in]
        :return: the updated h, x, v, and edge_attr
        """

        # aggregate force (equivariant message passing on the whole graph)
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, x)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [B*M, Ef], the global invariant message
        _, f = self.coord_model(x, edge_index, coord_diff, edge_feat)  # [B*N, 3]

        for type in cfg:
            x, v = self.update(x, v, f, h, node_index=cfg[type], type=type)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, others=h)

        return h, x, v, edge_attr

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