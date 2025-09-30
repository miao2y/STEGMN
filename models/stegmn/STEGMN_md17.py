from tkinter.filedialog import SaveAs
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from models.gcl import E_GCL_AT, E_GCL, GCL
from models.layer import AGLTSA
from models.stegmn.GMNLayer_Pooling import GMNLayer_Pooling
from models.stegmn.GMNLayer_AT import GMNLayer_AT
from papers.HEGNN.models.HEGNN import SH_INIT, HEGNN_Layer
from papers.HEGNN.models.md17.layer import GMNLayer
from transformer.Models import Encoder
from einops import rearrange


# Non-equivariant STAG
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, None, :]
        return self.dropout(x)

class STEGMN_MD17(nn.Module):
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
        tempo_attention= True,
        pooling=True
    ):
        super(STEGMN_MD17, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter
        self.tempo_attention = tempo_attention
        self.pooling = pooling

        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        
        # 初始化 embedding 权重，防止 NaN
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)

        if self.fft:
            in_edge_nf = num_past - 1
        self.sh_init = SH_INIT(
            edge_attr_dim=in_edge_nf,
            hidden_dim=hidden_nf,
            max_ell=6,
            activation=act_fn,
        )

        for i in range(n_layers):
            self.add_module(
                "egcl_%d" % (i * 2 + 1),
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    nodes_att_dim=nodes_att_dim,
                    act_fn=act_fn,
                    recurrent=True,
                    coords_weight=coords_weight,
                    norm_diff=True,
                    clamp=True,
                ),
            )
            self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
            self.add_module(
                "hegnn_%d" % (i * 2 + 1),
                HEGNN_Layer(
                    edge_attr_dim=in_edge_nf,
                    hidden_dim=hidden_nf,
                    sh_irreps=self.sh_init.sh_irreps,
                    activation=act_fn,
                ),
            )
            self.add_module(
                "gmn_%d" % (i * 2 + 1),
                GMNLayer(
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
                    learnable=True,
                ),
            )
            if self.eat:
                self.add_module(
                    "gmn_at_%d" % (i * 2 + 2),
                    GMNLayer_AT(
                        self.hidden_nf,
                        self.hidden_nf,
                        self.hidden_nf,
                        edges_in_d=in_edge_nf,
                        act_fn=act_fn,
                        recurrent=True,
                        coords_weight=coords_weight,
                        with_mask=with_mask,
                    ),
                )
        self.pooling = GMNLayer_Pooling(
                        self.hidden_nf,
                        self.hidden_nf,
                        self.hidden_nf,
                        edges_in_d=in_edge_nf,
                        act_fn=act_fn,
                        recurrent=True,
                        coords_weight=coords_weight,
                        with_mask=with_mask,
                    )
        self.theta = nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        self.theta.data.uniform_(-5, 5)
        # O init
        self.theta.data *= 0

    def forward(self, h, x, edges, edge_attr, node_vel, cfg):
        """parameters
        h: (b*n_node, 1)
        x: (num_past, b*n_node, 3)
        edges: (2, n_edge)
        edge_attr: (n_edge, 3)
        """
        # 检查输入是否包含 NaN
        if torch.isnan(h).any():
            print("Warning: Input h contains NaN values")
            h = torch.nan_to_num(h, nan=0.0)
        
        ### (num_past, b*n_node, hidden_nf)
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0], 1, 1))
        
        # 检查 embedding 输出是否包含 NaN
        if torch.isnan(h).any():
            print("Warning: Embedding output contains NaN values")
            h = torch.nan_to_num(h, nan=0.0)

        time_embedding = self.TimeEmbedding(
            torch.arange(self.num_past).to(self.device)
        ).unsqueeze(1)
        h = h + time_embedding
        # h = self.PosEmbedding(h)

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])

        # if Fs is not None:
        #     Fs = Fs.unsqueeze(1).repeat(1, h.shape[1], 1)

        h = rearrange(h, "bs_n seq h_dim -> (bs_n seq) h_dim", seq=self.num_past)
        x = rearrange(x, "bs_n seq x_dim -> (bs_n seq) x_dim", seq=self.num_past)
        node_vel = rearrange(
            node_vel, "seq bs_n x_dim -> (seq bs_n) x_dim", seq=self.num_past
        )

        for i in range(self.n_layers):

            # h, x, node_sh = self._modules["hegnn_%d" % (i * 2 + 1)](
            #     h, x, node_sh, node_vel, edges, edge_attr
            # )
            h, x, node_vel, _ = self._modules["gmn_%d" % (i * 2 + 1)](
                h, edges, x, node_vel, cfg, edge_attr=edge_attr
            )

            if self.eat:

                node_vel = rearrange(
                    node_vel, "(bs_n seq) x_dim -> seq bs_n x_dim", seq=self.num_past
                )
                
                x = rearrange(
                    x, " (bs_n seq) x_dim -> seq bs_n x_dim", seq=self.num_past
                )
                h = rearrange(
                    h, " (bs_n seq) x_dim -> seq bs_n x_dim", seq=self.num_past
                )

                if self.tempo_attention:
                    h, x, node_vel = self._modules["gmn_at_%d" % (i * 2 + 2)](h, edges, x, node_vel, cfg)
                else:
                    # print("start from here")
                    h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)


                h = rearrange(
                    h, "seq bs_n h_dim -> (bs_n seq) h_dim", seq=self.num_past
                )
                x = rearrange(
                    x, "seq bs_n x_dim -> (bs_n seq) x_dim", seq=self.num_past
                )
                node_vel = rearrange(
                    node_vel, "seq bs_n x_dim -> (bs_n seq) x_dim", seq=self.num_past
                )

        x = rearrange(x, " (bs_n seq) x_dim -> bs_n seq x_dim", seq=self.num_past)
        x = permute(x)
        node_vel = rearrange(node_vel, " (bs_n seq) x_dim -> bs_n seq x_dim", seq=self.num_past)
        node_vel = permute(node_vel)

        if(self.pooling):
            x_hat = self.pooling(h, edges, x, node_vel, cfg)
        else:
            # print("start from here")
            # 在时间的维度上取均值，作为 x_hat
            # x_hat = torch.mean(x, dim=0)  # [n_nodes, 3]
            x_hat = (
                torch.einsum("ij,jkt->ikt", self.theta, x - x[-1].unsqueeze(0)).squeeze(
                    0
                )
                + x[-1]
            )

        return x_hat

        # 写一个池化层
        # if self.tempo:
        #     x_hat = (
        #         torch.einsum("ij,jkt->ikt", self.theta, x - x[-1].unsqueeze(0)).squeeze(
        #             0
        #         )
        #         + x[-1]
        #     )
        # else:
        #     x_hat = torch.einsum(
        #         "ij,jkt->ikt", torch.softmax(self.theta, dim=1), x
        #     ).squeeze(0)

        return x_hat
