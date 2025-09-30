from torch import nn, optim
import torch

from models.stegmn.GMNLayerX import GMNLayerX
from models.stegmn.GMNLayerX_AT import GMNLayerX_AT
from models.stegmn.GMNLayerX_Pooling import GMNLayerX_Pooling
from models.stegmn.GMNLayerX_Pooling2 import GMNLayerX_Pooling2
from models.stegmn.GMNLayerX_Pooling3 import GMNLayerX_Pooling3
from models.stegmn.PositionalEncoding import PositionalEncoding
from papers.HEGNN.models.md17.layer import GMNLayer


class STEGMN(nn.Module):
    def __init__(
        self,
        num_past,
        num_future,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        fft,
        eat,
        device,
        n_layers,
        n_nodes,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        coords_weight=1.0,
        with_mask=False,
        tempo=True,
        filter=True,
    ):
        super(STEGMN, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter

        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.add_module("gmn_1", GMNLayerX(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf,
            edges_in_d=in_edge_nf,
            nodes_att_dim=0,
            act_fn=act_fn,
            coords_weight=coords_weight,
            recurrent=False,
            norm_diff=False,
            tanh=False,
            learnable=False,
        ))
        self.add_module("gmn_2", GMNLayerX(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf,
            edges_in_d=in_edge_nf,
            nodes_att_dim=0,
            act_fn=act_fn,
            coords_weight=coords_weight,
            recurrent=False,
            norm_diff=False,
            tanh=False,
            learnable=False,
        ))
        self.add_module("gmn_3", GMNLayerX(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf,
            edges_in_d=in_edge_nf,
            nodes_att_dim=0,
            act_fn=act_fn,
            coords_weight=coords_weight,
            recurrent=False,
            norm_diff=False,
            tanh=False,
            learnable=False,
        ))
        self.add_module("gmnt_1", GMNLayerX_AT(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf,
            edges_in_d=in_edge_nf,
            nodes_att_dim=0,
            act_fn=act_fn,
            coords_weight=coords_weight,
            recurrent=False,
            norm_diff=False,
            tanh=False,
            learnable=False,
        ))
        self.add_module("gmnt_2", GMNLayerX_AT(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf,
            edges_in_d=in_edge_nf,
            nodes_att_dim=0,
            act_fn=act_fn,
            coords_weight=coords_weight,
            recurrent=False,
            norm_diff=False,
            tanh=False,
            learnable=False,
        ))
        self.add_module("gmn_pooling", GMNLayerX_Pooling3(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf,
            edges_in_d=in_edge_nf,
            nodes_att_dim=0,
            act_fn=act_fn,
            coords_weight=coords_weight,
            recurrent=False,
            norm_diff=False,
            tanh=False,
            learnable=False,
            num_future=num_future,
            num_past=num_past,
        ))
        self.theta = nn.Parameter(torch.FloatTensor(num_future, num_past))


        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        self.theta.data.uniform_(-0.1, 0.1)
        # O init
        # self.theta.data *= 0

    def spatial_domain_forward(self, h, x, edges, vec, cfg,layer):
        """
        空间域模型前向传播
        对时间序列的每个时间步应用GMN层，并保存结果
        
        Args:
            h: 节点特征 [num_past, n_nodes, hidden_nf]
            x: 节点坐标 [num_past, n_nodes, 3]
            edges: 边索引 [2, n_edges]
            vec: 边特征 [num_past, n_edges, hidden_nf]
            cfg: 配置参数
            
        Returns:
            h: 处理后的节点特征 [num_past, n_nodes, hidden_nf]
            x: 处理后的节点坐标 [num_past, n_nodes, 3]
            vec: 处理后的边特征 [num_past, n_edges, hidden_nf]
        """
        # 初始化输出tensor
        h_output = torch.zeros_like(h)
        x_output = torch.zeros_like(x)
        vec_output = torch.zeros_like(vec)
        
        for i in range(h.shape[0]):
            h_t_out, x_t_out, vec_t_out, edge_attr_t = self._modules["gmn_%d" % layer](h[i], edges, x[i], vec[i], cfg)
            h_output[i] = h_t_out
            x_output[i] = x_t_out
            vec_output[i] = vec_t_out
        
        return h_output, x_output, vec_output
    
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
        
        # 应用时域pooling操作，参考公式：x_hat=torch.einsum("ij,jkts->ikts", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        
        # 计算相对于最后一个时间步的差值
        x_diff = x_isolated - x_isolated[-1].unsqueeze(0)  # [T, K, 4, 3]
        
        # 应用theta权重进行时域pooling，将时间维度从T压缩为1
        # self.theta: [num_future, num_past] = [1, T]
        # x_diff: [T, K, 4, 3]
        # 使用einsum: "ij,jkts->ikts" 其中 i=1, j=T, k=K, t=4, s=3
        x_diff_weighted = torch.einsum("ij,jkts->ikts", self.theta2, x_diff)  # [1, K, 4, 3]
        
        # 加上最后一个时间步的值，得到pooling后的结果
        x_isolated_pooled = x_diff_weighted.squeeze(0) + x_isolated[-1]  # [K, 4, 3]
        
        # 直接返回pooling后的结果
        return x_isolated_pooled, v[:, isolated_indices][-1]  # [K, 4, 3], [K, 4, 3]

    def forward(self, h, x, edges, edge_attr, vec, cfg):
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        # 空间域模型前向传播
        h, x, vec = self.spatial_domain_forward(h, x, edges, vec, cfg,1)

        # 时间域上的模型
        h, x, vec, edge_attr, score = self._modules["gmnt_1"](h, edges, x, vec, cfg)


        # 空间域模型前向传播
        h, x, vec = self.spatial_domain_forward(h, x, edges, vec, cfg,2)

        # 时间域上的模型
        h, x, vec, edge_attr, score = self._modules["gmnt_2"](h, edges, x, vec, cfg)

        x_hat = self._modules["gmn_pooling"](h, edges, x, vec, cfg)

        # x_hat=torch.einsum("ij,jkts->ikts", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]

        # x_hat = torch.einsum("ij,jkts->ikts", torch.softmax(self.theta,dim=1), x).squeeze(0)
        # 在时间的维度上取均值，作为 x_hat
        # x_hat = torch.mean(x, dim=0)  # [n_nodes, 3]
        # if x.shape[0]==1:
        #     x_hat=x.squeeze(0)
        # else:
        #     x_hat = torch.einsum("ij,jkts->ikts", torch.softmax(self.theta,dim=1), x).squeeze(0)
                # x_hat = torch.mean(x, dim=0)  # [n_nodes, 3]

        return x_hat

