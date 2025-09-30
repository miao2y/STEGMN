# 导入必要的库
import os
import random
import numpy as np
from scipy.sparse import coo_matrix
import torch

# from pytorch3d import transforms
from torch.utils.data import Dataset

from MDAnalysisData import datasets
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import distances


class MDAnalysisDataset(Dataset):
    """
    用于处理分子动力学模拟数据的PyTorch数据集类
    支持加载和处理MDAnalysis格式的分子轨迹数据
    """

    def __init__(
        self,
        dataset_name,  # 数据集名称
        partition="train",  # 数据集划分(train/valid/test)
        tmp_dir=None,  # 临时目录路径
        delta_frame=1,  # 帧间隔
        train_valid_test_ratio=None,  # 训练/验证/测试集比例
        test_rot=False,  # 是否测试旋转
        test_trans=False,  # 是否测试平移
        load_cached=False,  # 是否加载缓存数据
        cut_off=6,  # 截断距离
        num_past=10,  # 历史帧数
    ):
        super().__init__()
        self.delta_frame = delta_frame
        self.dataset = dataset_name
        self.partition = partition
        self.load_cached = load_cached
        self.test_rot = test_rot
        self.test_trans = test_trans
        self.cut_off = cut_off
        self.num_past = num_past
        if load_cached:
            print(f"Loading {dataset_name} from cached data for {partition}...")
            tmp_dir = os.path.join(tmp_dir, "adk_processed")
        self.tmp_dir = tmp_dir
        if train_valid_test_ratio is None:
            train_valid_test_ratio = [0.6, 0.2, 0.2]
        assert sum(train_valid_test_ratio) <= 1

        if load_cached:
            # 从缓存加载数据
            edges, self.edge_attr, self.charges, self.n_frames = torch.load(
                os.path.join(tmp_dir, f"{dataset_name}.pkl")
            )
            self.edges = torch.stack(edges, dim=0)
            self.train_valid_test = [
                int(train_valid_test_ratio[0] * (self.n_frames - delta_frame)),
                int(sum(train_valid_test_ratio[:2]) * (self.n_frames - delta_frame)),
            ]
            return

        if dataset_name.lower() == "adk":
            # 加载ADK数据集
            adk = datasets.fetch_adk_equilibrium(data_home=tmp_dir)
            
            # 创建MDAnalysis Universe对象
            self.data = mda.Universe(adk.topology, adk.trajectory)

            # print the number of atoms
            print(f"Number of atoms: {len(self.data.atoms)}")
            # print the number of residues
            print(f"Number of residues: {len(self.data.residues)}")
            # print the number of bonds
            print(f"Number of bonds: {len(self.data.bonds)}")
            # print the number of bonds
            # 选择骨架原子，只考虑骨架上的原子
            backbone = self.data.atoms.select_atoms("backbone")
            bb_by_res = backbone.split("residue")

            # 筛选每个残基包含4个原子的骨架
            # 因为标准的蛋白质骨架原子应该包含4个原子：N、CA、C、O
            # 去掉了蛋白质末端残基
            self.bb_by_res = [x for x in bb_by_res if len(x) == 4]
            
            # 并不是数学上的加法运算，是用于将多个残基的骨架原子合并成一个整体
            self.backbone = sum(self.bb_by_res)

            # 选择CA原子
            # 在蛋白质结构中，CA原子通常用来表示整个残基的位置
            self.CA = self.backbone.select_atoms("name CA")
            self.id_ca = dict(zip(self.CA.ids, list(range(len(self.CA)))))
            '''
                形如：
                    402: 0,
                    406: 1,
                    410: 2,
                    414: 3
            '''

        else:
            raise NotImplementedError(
                f"{dataset_name} is not available in MDAnalysisData."
            )

        # 构建局部图信息
        try:
            # 获取原子电荷
            self.charges = torch.stack(
                [torch.tensor(bb.charges) for bb in self.bb_by_res]
            )
        except OSError:
            print(f"Charge error")
        try:
            # 获取键连接信息
            self.edges = torch.stack(
                [
                    torch.tensor(self.CA.bonds.indices[:, 0], dtype=torch.long),
                    torch.tensor(self.CA.bonds.indices[:, 1], dtype=torch.long),
                ],
                dim=0,
            )
        except OSError:
            print(f"edges error")
        try:
            # 获取键长信息
            self.edge_attr = torch.tensor(
                [bond.length() for bond in self.CA.bonds]
            ).reshape(-1, 1)
        except OSError:
            print(f"edge_attr error")

        # 计算训练/验证/测试集划分点
        self.train_valid_test = [
            int(
                train_valid_test_ratio[0]
                * (len(self.data.trajectory) - self.delta_frame * self.num_past)
            ),
            int(
                sum(train_valid_test_ratio[:2])
                * (len(self.data.trajectory) - self.delta_frame * self.num_past)
            ),
        ]

        # 获取所有帧的原子位置
        x = torch.tensor(
            np.stack(
                [
                    self.data.trajectory[t].positions
                    for t in range(len(self.data.trajectory))
                ]
            )
        )
        print(x.shape)
        # x shape is [4187, 3341, 3])
        # 4187是帧数，3341是原子数，3是维度（x,y,z）

        x = x[:-1]
        # 去掉最后一帧
        # x shape is [4186, 3341, 3]

        # 提取骨架原子位置
        # torch.Size([4186, 213, 4, 3])
        # 4186 是帧数，213 是残基数，4 是每个残基包含的原子数，3是维度（x,y,z）
        self.X_bb = torch.stack([x[:, bb.ids, :] for bb in self.bb_by_res], axis=1)

        # 计算每个原子的速度
        self.V_bb = torch.diff(self.X_bb, dim=0)
        # 第一帧的速度复制了一遍
        self.V_bb = torch.cat([self.V_bb[0:1], self.V_bb], dim=0) 
        # 获取第一帧的原子位置
        # shape is (3341, 3)
        # 3341是所有原子数，3是维度（x,y,z）
        x_0 = np.ascontiguousarray(self.data.trajectory[0].positions)

        # 获取第一帧的CA原子位置
        # shape is (213, 3)
        # 213是CA原子数，3是维度（x,y,z）
        x_0_ca = x_0[self.CA.ids]

        # 构建全局接触矩阵
        # 小于 self.cut_off 的距离被视为接触
        # coo_matrix: 将结果转换为稀疏矩阵
        edge_global = coo_matrix(
            distances.contact_matrix(x_0_ca, cutoff=self.cut_off, returntype="sparse")
        )
        #  将对角线设为0
        edge_global.setdiag(False)
        # 删除所有值为0的元素
        edge_global.eliminate_zeros()

        # 将稀疏矩阵转换为张量
        # 1126 条边，213 个 CA 原子
        self.edge_global = torch.stack(
            [
                torch.tensor(edge_global.row, dtype=torch.long),
                torch.tensor(edge_global.col, dtype=torch.long),
            ],
            dim=0,
        )

        # 计算全局边的属性(距离)
        self.edge_global_attr = torch.norm(
            torch.tensor(x_0_ca)[self.edge_global[0], :]
            - torch.tensor(x_0_ca)[self.edge_global[1], :],
            p=2,
            dim=1,
        ).unsqueeze(-1)

        
        selected_edges, selected_indices, unused_vertices = self.build_sticks(
            self.edge_global, 
            len(self.id_ca),
            # max_edges=50, 
            seed=42
        )
        # 构建 config 字典
        # {'Stick': [(0, 1), (2, 3), (4, 5)], 'Isolated': [[6], [7], [8], [9], [10], [11]]}
        # selected_edges 改为 Stick
        edges = selected_edges.cpu().numpy().T
    
        # 将每条边转换为元组，并放入列表中
        stick_edges = [(int(v1), int(v2)) for v1, v2 in edges]

        vertices = unused_vertices.cpu().numpy()
    
        # 将每个顶点转换为单元素列表，并放入列表中
        isolated_vertices = [[int(v)] for v in vertices]
        
        # 返回字典格式
            
        # 返回字典格式
        self.cfg = {'Stick': stick_edges, 'Isolated': isolated_vertices}
       


        """
        edge_attrs, Fss = [], []
        for i in range(self.X_bb.shape[0]-self.delta_frame*self.num_past):
            print(i)
            edge_attr, Fs = FFT(self.X_bb[[i+k*self.delta_frame for k in range(self.num_past)],:,1,:], len(self.CA),1, edges=self.edge_global)
            edge_attrs.append(edge_attr)
            Fss.append(Fs)
        edge_attrs_=torch.stack(edge_attrs)
        Fss_=torch.stack(Fss)
        torch.save(edge_attrs_,'mdanalysis/edge_attr_fft.pt')
        torch.save(Fss_,'mdanalysis/Fs_fft.pt')
        """

        self.edge_attr_fft = torch.load("mdanalysis/edge_attr_fft.pt")
        self.Fs_fft = torch.load("mdanalysis/Fs_fft.pt")

        # 构建邻接矩阵
        # 初始化一个全0的邻接矩阵，大小为 (原子数, 原子数)
        self.A = torch.zeros(self.charges.shape[0], self.charges.shape[0])
        for i in range(self.edge_global.shape[1]):
            self.A[self.edge_global[0, i], self.edge_global[1, i]] = (
                self.edge_global_attr[i]
            )

        # 归一化邻接矩阵
        self.A = get_normalized_adj(self.A)

    def __getitem__(self, i):
        """
        获取数据集中的一个样本
        返回: 初始位置、边属性、电荷、目标位置、FFT边属性、FFT特征
        """
        charges, edge_attr = self.charges, self.edge_global_attr
        if len(charges.size()) == 1:
            charges = charges.unsqueeze(-1)
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # 根据数据集划分调整索引
        if self.partition == "valid":
            i = i + self.train_valid_test[0]
        elif self.partition == "test":
            i = i + self.train_valid_test[1]

        # 获取历史帧和目标帧
        # 历史帧是前10帧（self.num_past）
        # 目标帧是第11帧
        frame_0, frame_t = [
            i + k * self.delta_frame for k in range(self.num_past)
        ], i + self.delta_frame * self.num_past

        # 返回历史帧（主干帧）、边属性（全局边属性）、电荷、目标帧、FFT边属性、FFT特征
        # !!! 原子数是 213
        # 历史帧/速度：[10, 213, 4, 3]，即历史长度*原子数*每个原子包含的原子数*维度
        # 1126 条边，edge_attr 是 1126 个值，每个值是 1 个数
        cfg = self.cfg
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}
        return (
            self.X_bb[frame_0],
            edge_attr,
            charges,
            self.X_bb[frame_t],
            self.edge_attr_fft[i],
            self.Fs_fft[i],
            self.V_bb[frame_0],
            cfg,
        )

    def __len__(self):
        """
        返回数据集长度
        """
        if self.load_cached:
            total_len = max(0, self.n_frames - self.delta_frame)
        else:
            total_len = max(
                0, len(self.data.trajectory) - self.delta_frame * self.num_past - 1
            )
        if self.partition == "train":
            return min(total_len, self.train_valid_test[0])
        if self.partition == "valid":
            return max(
                0, min(total_len, self.train_valid_test[1]) - self.train_valid_test[0]
            )
        if self.partition == "test":
            return max(0, total_len - self.train_valid_test[1])
        

    def build_sticks(self, edge_global,nodes, max_edges=None, seed=42):
        """∂
        从边集合中选择不重叠的边（边的顶点不重复），并返回未选中的顶点
        
        参数:
            edge_global: torch.Tensor, 形状为 [2, num_edges] 的边集合
            max_edges: int, 最多选择的边数，如果为None则不限制
            seed: int, 随机种子，用于控制随机性
        
        返回:
            selected_edges: torch.Tensor, 形状为 [2, num_selected_edges] 的选中边集合
            selected_indices: torch.Tensor, 选中边的索引
            unused_vertices: torch.Tensor, 未选中的顶点索引
        """
        # 设置随机种子
        np.random.seed(seed)
        
        # 转换为CPU并转为numpy以便处理
        edges = edge_global.cpu().numpy()
        
        # 记录已使用的顶点
        used_vertices = set()
        selected_indices = []
        
        # 随机打乱边的顺序
        edge_indices = np.random.permutation(edges.shape[1])
        
        # 遍历所有边
        for idx in edge_indices:
            # 如果达到最大边数限制，则停止
            if max_edges is not None and len(selected_indices) >= max_edges:
                break
            
            v1, v2 = edges[:, idx]
            
            # 如果两个顶点都未被使用，则选择这条边
            if v1 not in used_vertices and v2 not in used_vertices:
                selected_indices.append(idx)
                used_vertices.add(v1)
                used_vertices.add(v2)
        
        # 转换回tensor
        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        selected_edges = edge_global[:, selected_indices]
        
        # 找出所有顶点
        all_vertices = set(range(nodes))
        
        # 找出未使用的顶点
        unused_vertices = torch.tensor(list(all_vertices - used_vertices), dtype=torch.long)
        
        return selected_edges, selected_indices, unused_vertices

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (
                index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)
            ).reshape(-1, index.shape[-1])
            if type == "Isolated":
                cfg[type] = cfg[type].squeeze(-1)
        return cfg

    def get_edges(self, batch_size, n_nodes):
        edges = [
            torch.LongTensor(self.edge_global[0]),
            torch.LongTensor(self.edge_global[1]),
        ]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges
    
    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (
                index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)
            ).reshape(-1, index.shape[-1])
            if type == "Isolated":
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


def collate_mda(data):
    loc_0, edge_attr, charges, loc_t, edge_attr_fft, Fs_fft, vec, cfg = zip(*data)

    edge_attr = torch.cat(edge_attr, dim=0).type(torch.float)
    loc_0 = torch.cat(loc_0, axis=1).type(torch.float)
    vec = torch.cat(vec, axis=1).type(torch.float)
    loc_t = torch.cat(loc_t, axis=0).type(torch.float)
    charges = torch.cat(charges, dim=0).type(torch.float)
    edge_attr_fft = torch.cat(edge_attr_fft, dim=0).type(torch.float)
    Fs_fft = torch.cat(Fs_fft, dim=0).type(torch.float)

    # 合并cfg为一个字典，每个key对应二维数组
    merged_cfg = {}
    for key in cfg[0].keys():
        merged_cfg[key] = torch.stack([c[key] for c in cfg], dim=0)

    return loc_0, edge_attr, charges, loc_t, edge_attr_fft, Fs_fft, vec, merged_cfg


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    # A = A + torch.diag(torch.ones(A.shape[0], dtype=torch.float32))
    A_ = torch.tensor(A, dtype=torch.float32)
    D = torch.sum(A_, axis=1)
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = torch.reciprocal(torch.sqrt(D))
    A_wave = torch.multiply(
        torch.multiply(diag.reshape((-1, 1)), A_), diag.reshape((1, -1))
    )
    return A_wave


def FFT(x, n_nodes, batch_size, edges):
    x_bar = torch.cat(
        [
            torch.mean(x[:, i * n_nodes : (i + 1) * n_nodes, :], axis=1)
            .unsqueeze(1)
            .repeat(1, n_nodes, 1)
            for i in range(batch_size)
        ],
        axis=1,
    )
    x_norm = x - x_bar
    F = torch.stack(
        [
            torch.fft.fft(x_norm[:, i, j])
            for i in range(x_norm.shape[1])
            for j in range(x_norm.shape[2])
        ],
        axis=1,
    ).view(x.shape)
    # A=torch.stack([torch.stack([cal_similarity_fourier(F[j,n_nodes*i:n_nodes*(i+1),:]) for i in range(batch_size)]) for j in range(x.shape[0])],axis=-1)

    Fs = torch.abs(torch.einsum("ijt,ijt->ij", F, F))[1:, :].T
    Fs_norm = Fs / torch.sum(Fs, axis=1).unsqueeze(-1)

    # edge_attr=torch.stack([A[edges[0][i].item()//A.shape[1]][edges[0][i].item()%A.shape[1]][edges[1][i].item()%A.shape[1]] for i in range(len(edges[0]))])
    edge_attr = torch.stack(
        [
            torch.abs(
                torch.sum(
                    torch.conj(F[:, edges[0][i], :]) * F[:, edges[1][i], :], axis=1
                )
            )
            for i in range(len(edges[0]))
        ]
    )
    edge_attr = edge_attr[:, 1:]
    edge_attr_norm = edge_attr / torch.sum(edge_attr, axis=1).unsqueeze(-1)

    return edge_attr_norm, Fs_norm
