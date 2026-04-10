#!/usr/bin/env python3
"""
MediaPipe 33 點 One-Shot 動作辨識模型

實作論文「One-Shot Action Recognition via Multi-Scale Spatial-Temporal
Skeleton Matching」，適配為 MediaPipe 33 關節格式。

主要元件：
- MediaPipe 33 點圖結構定義
- 自適應圖卷積網路 (AGCN)
- Earth Mover's Distance (EMD) 最佳匹配
- 多尺度時空匹配

Author: Claude AI Assistant
Date: 2026-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import ot  # Python Optimal Transport 函式庫
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("警告：找不到 'pot' 函式庫，將使用簡化版 EMD。")


# =============================================================================
# MediaPipe 33 關節圖定義
# =============================================================================

class MediaPipeGraph:
    """
    MediaPipe Pose 33 關鍵點的圖結構。

    關鍵點索引：
    0: 鼻子
    1: 左眼內    2: 左眼      3: 左眼外
    4: 右眼內    5: 右眼      6: 右眼外
    7: 左耳      8: 右耳
    9: 嘴巴左   10: 嘴巴右
    11: 左肩    12: 右肩
    13: 左肘    14: 右肘
    15: 左腕    16: 右腕
    17: 左小指  18: 右小指
    19: 左食指  20: 右食指
    21: 左拇指  22: 右拇指
    23: 左髖    24: 右髖
    25: 左膝    26: 右膝
    27: 左踝    28: 右踝
    29: 左腳跟  30: 右腳跟
    31: 左腳尖  32: 右腳尖
    """

    def __init__(self):
        self.num_joints = 33

        # 定義邊（骨架連接）- 基於 MediaPipe Pose 的官方連接
        self.edges = [
            # 臉部
            (0, 1), (1, 2), (2, 3), (3, 7),    # 左眼到左耳
            (0, 4), (4, 5), (5, 6), (6, 8),    # 右眼到右耳
            (9, 10),                            # 嘴巴

            # 軀幹
            (11, 12),                           # 肩膀連線
            (11, 23), (12, 24),                 # 肩膀到髖部
            (23, 24),                           # 髖部連線

            # 左臂
            (11, 13), (13, 15),                 # 肩膀-肘-腕
            (15, 17), (15, 19), (15, 21),       # 腕到手指
            (17, 19),                           # 小指到食指

            # 右臂
            (12, 14), (14, 16),                 # 肩膀-肘-腕
            (16, 18), (16, 20), (16, 22),       # 腕到手指
            (18, 20),                           # 小指到食指

            # 左腿
            (23, 25), (25, 27),                 # 髖-膝-踝
            (27, 29), (27, 31), (29, 31),       # 踝-腳跟-腳尖

            # 右腿
            (24, 26), (26, 28),                 # 髖-膝-踝
            (28, 30), (28, 32), (30, 32),       # 踝-腳跟-腳尖
        ]

        # 中心關節（用於空間配置）
        self.center = 0  # 鼻子

        # =================================================================
        # 多尺度空間池化群組
        # =================================================================

        # 尺度 2：12 部位 (Scale 2: 12 parts)
        self.scale2_groups = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 頭部 (11點)
            1: [11, 12, 23, 24],                     # 軀幹 (4點)
            2: [11, 13],                             # 左上臂 (2點)
            3: [13, 15],                             # 左前臂 (2點)
            4: [15, 17, 19, 21],                     # 左手 (4點)
            5: [12, 14],                             # 右上臂 (2點)
            6: [14, 16],                             # 右前臂 (2點)
            7: [16, 18, 20, 22],                     # 右手 (4點)
            8: [23, 25],                             # 左大腿 (2點)
            9: [25, 27, 29, 31],                     # 左小腿+腳 (4點)
            10: [24, 26],                            # 右大腿 (2點)
            11: [26, 28, 30, 32],                    # 右小腿+腳 (4點)
        }

        # 尺度 3：6 超級部位 (Scale 3: 6 super-parts)
        self.scale3_groups = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # 頭部
            1: [11, 12, 23, 24],                      # 軀幹
            2: [11, 13, 15, 17, 19, 21],              # 左臂（含手）
            3: [12, 14, 16, 18, 20, 22],              # 右臂（含手）
            4: [23, 25, 27, 29, 31],                  # 左腿（含腳）
            5: [24, 26, 28, 30, 32],                  # 右腿（含腳）
        }

    def get_adjacency(self):
        """取得圖的鄰接矩陣（33x33）。"""
        A = np.zeros((self.num_joints, self.num_joints), dtype=np.float32)

        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1

        # 加入自環
        A = A + np.eye(self.num_joints, dtype=np.float32)

        return A

    def get_scale2_adjacency(self):
        """取得尺度 2（12 部位）的鄰接矩陣。"""
        num_parts = len(self.scale2_groups)
        A = np.zeros((num_parts, num_parts), dtype=np.float32)

        # 定義部位間的連接
        edges = [
            (0, 1),              # 頭部 - 軀幹
            (1, 2), (1, 5),      # 軀幹 - 上臂
            (2, 3), (3, 4),      # 左臂鏈
            (5, 6), (6, 7),      # 右臂鏈
            (1, 8), (1, 10),     # 軀幹 - 大腿
            (8, 9), (10, 11),    # 腿鏈
        ]

        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1

        A = A + np.eye(num_parts, dtype=np.float32)
        return A

    def get_scale3_adjacency(self):
        """取得尺度 3（6 超級部位）的鄰接矩陣。"""
        num_parts = len(self.scale3_groups)
        A = np.zeros((num_parts, num_parts), dtype=np.float32)

        # 定義連接
        edges = [
            (0, 1),              # 頭部 - 軀幹
            (1, 2), (1, 3),      # 軀幹 - 手臂
            (1, 4), (1, 5),      # 軀幹 - 腿
        ]

        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1

        A = A + np.eye(num_parts, dtype=np.float32)
        return A


# =============================================================================
# 圖卷積層
# =============================================================================

class GraphConv(nn.Module):
    """基本圖卷積層。"""

    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 鄰接矩陣
        self.register_buffer('A', torch.from_numpy(A))
        self.num_nodes = A.shape[0]

        # 卷積權重
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 自適應鄰接
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.zeros_like(self.A))
            self.alpha = nn.Parameter(torch.zeros(1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        參數：
            x: (N, C, T, V) - 批次, 通道, 時間, 頂點
        """
        # 取得鄰接矩陣
        if self.adaptive:
            A = self.A + self.PA * self.alpha
        else:
            A = self.A

        # 正規化鄰接矩陣
        D = torch.sum(A, dim=1, keepdim=True)
        A = A / (D + 1e-6)

        # 圖卷積：x @ A
        # x: (N, C, T, V), A: (V, V)
        x = torch.einsum('nctv,vw->nctw', x, A)

        # 通道混合
        x = self.conv(x)
        x = self.bn(x)

        return x


class TemporalConv(nn.Module):
    """時間卷積層。"""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class AGCBlock(nn.Module):
    """自適應圖卷積區塊。"""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()

        self.gcn = GraphConv(in_channels, out_channels, A, adaptive=True)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        # 殘差連接
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.relu(x + res)
        return x


# =============================================================================
# 嵌入網路
# =============================================================================

class MediaPipeSkeletonEmbedding(nn.Module):
    """
    使用 AGCN 的多尺度骨架嵌入網路（MediaPipe 33 點版本）。

    架構：
    - 6 個共享 AGC 區塊
    - 每個空間尺度各有 3 個獨立區塊
    """

    def __init__(self, in_channels=3, base_channels=64, num_classes=None):
        super().__init__()

        self.graph = MediaPipeGraph()

        # 取得鄰接矩陣
        A1 = self.graph.get_adjacency()           # 33x33
        A2 = self.graph.get_scale2_adjacency()    # 12x12
        A3 = self.graph.get_scale3_adjacency()    # 6x6

        # 輸入投影
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_joints)

        # 共享區塊 (1-6)
        self.shared_blocks = nn.ModuleList([
            AGCBlock(in_channels, base_channels, A1, residual=False),
            AGCBlock(base_channels, base_channels, A1),
            AGCBlock(base_channels, base_channels, A1),
            AGCBlock(base_channels, base_channels * 2, A1, stride=2),
            AGCBlock(base_channels * 2, base_channels * 2, A1),
            AGCBlock(base_channels * 2, base_channels * 2, A1),
        ])

        # 尺度 1 專用區塊（33 關節）
        self.scale1_blocks = nn.ModuleList([
            AGCBlock(base_channels * 2, base_channels * 4, A1, stride=2),
            AGCBlock(base_channels * 4, base_channels * 4, A1),
            AGCBlock(base_channels * 4, base_channels * 4, A1),
        ])

        # 尺度 2 專用區塊（12 部位）
        self.scale2_blocks = nn.ModuleList([
            AGCBlock(base_channels * 2, base_channels * 4, A2, stride=2),
            AGCBlock(base_channels * 4, base_channels * 4, A2),
            AGCBlock(base_channels * 4, base_channels * 4, A2),
        ])

        # 尺度 3 專用區塊（6 超級部位）
        self.scale3_blocks = nn.ModuleList([
            AGCBlock(base_channels * 2, base_channels * 4, A3, stride=2),
            AGCBlock(base_channels * 4, base_channels * 4, A3),
            AGCBlock(base_channels * 4, base_channels * 4, A3),
        ])

        self.out_channels = base_channels * 4

        # 可選的分類器（用於預訓練）
        if num_classes is not None:
            self.classifier = nn.Linear(base_channels * 4, num_classes)
        else:
            self.classifier = None

    def spatial_pool(self, x, groups):
        """
        根據空間群組池化特徵。

        參數：
            x: (N, C, T, V)
            groups: 將 group_id 映射到關節索引列表的字典

        回傳：
            池化後的特徵 (N, C, T, num_groups)
        """
        N, C, T, V = x.shape
        num_groups = len(groups)
        pooled = torch.zeros(N, C, T, num_groups, device=x.device)

        for g_id, joints in groups.items():
            pooled[:, :, :, g_id] = x[:, :, :, joints].mean(dim=-1)

        return pooled

    def forward(self, x, return_multi_scale=False):
        """
        前向傳播。

        參數：
            x: 輸入張量 (N, C, T, V) 或 (N, T, V, C)
            return_multi_scale: 若為 True，回傳所有尺度的特徵

        回傳：
            若 return_multi_scale:
                (scale1_feat, scale2_feat, scale3_feat)
            否則:
                全域池化後的特徵
        """
        # 輸入形狀處理
        if x.dim() == 4 and x.shape[-1] == 3:
            # (N, T, V, C) -> (N, C, T, V)
            x = x.permute(0, 3, 1, 2)

        N, C, T, V = x.shape

        # 批次正規化
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        # 共享區塊
        for block in self.shared_blocks:
            x = block(x)

        # 尺度 1（33 關節）
        x1 = x
        for block in self.scale1_blocks:
            x1 = block(x1)

        # 尺度 2（12 部位）- 先池化再處理
        x2 = self.spatial_pool(x, self.graph.scale2_groups)
        for block in self.scale2_blocks:
            x2 = block(x2)

        # 尺度 3（6 超級部位）
        x3 = self.spatial_pool(x, self.graph.scale3_groups)
        for block in self.scale3_blocks:
            x3 = block(x3)

        if return_multi_scale:
            return x1, x2, x3

        # 用於分類的全域池化
        x1 = F.adaptive_avg_pool2d(x1, 1).squeeze(-1).squeeze(-1)

        if self.classifier is not None:
            return self.classifier(x1)

        return x1


# =============================================================================
# Earth Mover's Distance (EMD) 匹配
# =============================================================================

class EMDMatcher:
    """
    基於 Earth Mover's Distance 的最佳匹配。

    實作論文中的匹配策略。
    """

    def __init__(self, use_pot=True):
        self.use_pot = use_pot and HAS_POT

    def compute_distance_matrix(self, X, Y):
        """
        計算成對餘弦距離矩陣。

        參數：
            X: (C, M) 特徵矩陣（M 個節點）
            Y: (C, N) 特徵矩陣（N 個節點）

        回傳：
            距離矩陣 (M, N)
        """
        # 正規化
        X_norm = F.normalize(X, p=2, dim=0)
        Y_norm = F.normalize(Y, p=2, dim=0)

        # 餘弦相似度 -> 距離
        similarity = torch.mm(X_norm.t(), Y_norm)  # (M, N)
        distance = 1 - similarity

        return distance

    def compute_weights(self, X, Y):
        """
        使用交叉參考機制計算節點權重（公式 4）。

        參數：
            X: (C, M) 特徵
            Y: (C, N) 特徵

        回傳：
            r: X 的權重 (M,)
            c: Y 的權重 (N,)
        """
        # 平均特徵
        Y_mean = Y.mean(dim=1, keepdim=True)  # (C, 1)
        X_mean = X.mean(dim=1, keepdim=True)

        # 計算權重
        r = torch.mm(X.t(), Y_mean).squeeze()  # (M,)
        c = torch.mm(Y.t(), X_mean).squeeze()  # (N,)

        # 確保非負
        r = F.relu(r) + 1e-6
        c = F.relu(c) + 1e-6

        # 正規化使總和為 1
        r = r / r.sum()
        c = c / c.sum()

        return r, c

    def compute_emd(self, X, Y):
        """
        計算兩個特徵集之間的 EMD。

        參數：
            X: (C, M) 或 (N, C, T, V)
            Y: (C, N) 或 (N, C, T, V)

        回傳：
            EMD 距離和語義相關性分數
        """
        # 如需要則展平
        if X.dim() == 4:
            N, C, T, V = X.shape
            X = X.view(C, -1)  # (C, T*V)
        if Y.dim() == 4:
            Y = Y.view(Y.shape[1], -1)

        # 計算距離矩陣
        D = self.compute_distance_matrix(X, Y)

        # 計算權重
        r, c = self.compute_weights(X, Y)

        if self.use_pot:
            # 使用 POT 函式庫計算精確 EMD
            r_np = r.detach().cpu().numpy().astype(np.float64)
            c_np = c.detach().cpu().numpy().astype(np.float64)
            D_np = D.detach().cpu().numpy().astype(np.float64)

            # 計算最佳傳輸計畫
            pi = ot.emd(r_np, c_np, D_np)
            pi = torch.from_numpy(pi).to(X.device).float()
        else:
            # 簡化版：使用 Sinkhorn
            pi = self.sinkhorn(r, c, D)

        # 計算語義相關性分數（公式 5）
        similarity = 1 - D
        score = (similarity * pi).sum()

        # EMD 距離
        emd = (D * pi).sum()

        return emd, score

    def sinkhorn(self, r, c, D, reg=0.1, max_iter=100):
        """
        Sinkhorn-Knopp 演算法用於近似最佳傳輸。
        """
        K = torch.exp(-D / reg)
        u = torch.ones_like(r)
        v = torch.ones_like(c)

        for _ in range(max_iter):
            u = r / (K @ v + 1e-8)
            v = c / (K.t() @ u + 1e-8)

        pi = torch.diag(u) @ K @ torch.diag(v)
        return pi


# =============================================================================
# 多尺度匹配模組
# =============================================================================

class MultiScaleMatcher(nn.Module):
    """
    用於 one-shot 動作辨識的多尺度與跨尺度匹配。
    """

    def __init__(self):
        super().__init__()
        self.emd = EMDMatcher()

    def temporal_pool(self, x, scale):
        """
        池化時間維度以建立較粗的尺度。

        參數：
            x: (N, C, T, V)
            scale: 1, 2, 或 4

        回傳：
            池化後的特徵
        """
        if scale == 1:
            return x

        return F.avg_pool2d(x, kernel_size=(scale, 1), stride=(scale, 1))

    def compute_multi_scale_score(self, X_scales, Y_scales):
        """
        計算多尺度匹配分數（公式 6, 7）。

        參數：
            X_scales: (x_s1, x_s2, x_s3) 空間尺度列表
            Y_scales: (y_s1, y_s2, y_s3) 空間尺度列表

        回傳：
            總語義相關性分數
        """
        total_score = 0

        for x, y in zip(X_scales, Y_scales):
            _, score = self.emd.compute_emd(x, y)
            total_score += score

        return total_score

    def compute_multi_temporal_score(self, X, Y):
        """
        計算多時間尺度匹配分數（公式 7）。

        參數：
            X, Y: (N, C, T, V)

        回傳：
            跨時間尺度的總分數
        """
        total_score = 0

        for scale in [1, 2, 4]:
            x_t = self.temporal_pool(X, scale)
            y_t = self.temporal_pool(Y, scale)
            _, score = self.emd.compute_emd(x_t, y_t)
            total_score += score

        return total_score

    def compute_cross_scale_score(self, X_scales, Y_scales):
        """
        計算跨尺度匹配分數（公式 8, 9）。

        匹配 X 和 Y 之間的不同尺度以處理不同的動作幅度。
        """
        total_score = 0

        # 跨空間尺度
        for i in range(len(X_scales)):
            for j in range(len(Y_scales)):
                if i != j:
                    # 池化至相同形狀（時間池化）
                    x = F.adaptive_avg_pool2d(X_scales[i], (1, 1))
                    y = F.adaptive_avg_pool2d(Y_scales[j], (1, 1))
                    x = x.view(x.shape[0], x.shape[1], -1)
                    y = y.view(y.shape[0], y.shape[1], -1)
                    _, score = self.emd.compute_emd(x, y)
                    total_score += score

        return total_score

    def forward(self, X_features, Y_features):
        """
        計算查詢和支持之間的總匹配分數。

        參數：
            X_features: (scale1, scale2, scale3) 特徵元組
            Y_features: (scale1, scale2, scale3) 特徵元組

        回傳：
            總語義相關性分數
        """
        # 多空間尺度匹配
        ms_score = self.compute_multi_scale_score(X_features, Y_features)

        # 多時間尺度匹配（在尺度 1 上）
        mt_score = self.compute_multi_temporal_score(
            X_features[0], Y_features[0]
        )

        # 跨尺度匹配
        cs_score = self.compute_cross_scale_score(X_features, Y_features)

        # 組合分數
        total_score = ms_score + mt_score + cs_score

        return total_score


# =============================================================================
# One-Shot 動作辨識模型
# =============================================================================

class OneShotActionRecognitionMediaPipe(nn.Module):
    """
    完整的 one-shot 動作辨識模型（MediaPipe 33 點版本）。
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.embedding = MediaPipeSkeletonEmbedding(
            in_channels=in_channels,
            base_channels=base_channels
        )
        self.matcher = MultiScaleMatcher()

    def extract_features(self, x):
        """
        從骨架序列中提取多尺度特徵。

        參數：
            x: (N, T, V, C) 骨架序列，V=33 for MediaPipe

        回傳：
            不同尺度的特徵元組
        """
        return self.embedding(x, return_multi_scale=True)

    def compute_similarity(self, query, support):
        """
        計算查詢和支持序列之間的相似度。

        參數：
            query: (1, T, V, C) 查詢骨架序列
            support: (1, T, V, C) 支持骨架序列

        回傳：
            相似度分數
        """
        # 提取特徵
        query_features = self.extract_features(query)
        support_features = self.extract_features(support)

        # 計算匹配分數
        score = self.matcher(query_features, support_features)

        return score

    def forward(self, query, support_set):
        """
        執行 one-shot 動作辨識。

        參數：
            query: (1, T, V, C) 查詢骨架序列
            support_set: (sequence, label) 元組列表

        回傳：
            預測的標籤和每個類別的分數
        """
        scores = []

        for support_seq, label in support_set:
            score = self.compute_similarity(query, support_seq)
            scores.append((score.item(), label))

        # 依分數排序（越高越好）
        scores.sort(key=lambda x: x[0], reverse=True)

        return scores[0][1], scores


# =============================================================================
# 工具函數
# =============================================================================

def preprocess_mediapipe_skeleton(keypoints, target_length=64):
    """
    預處理 MediaPipe 骨架序列作為模型輸入。

    參數：
        keypoints: (T, 33, C) 原始關鍵點，C=3 (x, y, z) 或 C=4 (x, y, z, visibility)
        target_length: 目標序列長度

    回傳：
        預處理後的張量 (1, T, 33, 3)
    """
    T, N, C = keypoints.shape

    # 確保只取前 3 個通道 (x, y, z)
    if C > 3:
        keypoints = keypoints[:, :, :3]

    # 時間插值/取樣
    if T != target_length:
        indices = np.linspace(0, T - 1, target_length).astype(int)
        keypoints = keypoints[indices]

    # 正規化座標（以髖部中心為原點）
    # MediaPipe: 23=左髖, 24=右髖
    center = (keypoints[:, 23, :2] + keypoints[:, 24, :2]) / 2
    keypoints[:, :, :2] -= center[:, np.newaxis, :]

    # 轉換為張量
    tensor = torch.from_numpy(keypoints).float().unsqueeze(0)

    return tensor


# =============================================================================
# NTU 到 MediaPipe 映射（備用方案）
# =============================================================================

def ntu_to_mediapipe_skeleton(ntu_skeleton):
    """
    將 NTU RGB+D 25 關節骨架轉換為 MediaPipe 33 關節格式。

    注意：這會有 14 個關節缺失，僅作為備用方案。
    建議使用 extract_mediapipe_from_ntu.py 重新從影片提取。

    參數：
        ntu_skeleton: (T, 25, 3)

    回傳：
        mediapipe_skeleton: (T, 33, 3)
    """
    T = ntu_skeleton.shape[0]
    mp_skeleton = np.zeros((T, 33, 3), dtype=np.float32)

    # NTU 到 MediaPipe 映射表
    mapping = {
        0: 3,    # Nose ← Head
        11: 4,   # Left Shoulder ← Shoulder Left
        12: 8,   # Right Shoulder ← Shoulder Right
        13: 5,   # Left Elbow ← Elbow Left
        14: 9,   # Right Elbow ← Elbow Right
        15: 6,   # Left Wrist ← Wrist Left
        16: 10,  # Right Wrist ← Wrist Right
        19: 21,  # Left Index ← Hand Tip Left
        20: 23,  # Right Index ← Hand Tip Right
        21: 22,  # Left Thumb ← Thumb Left
        22: 24,  # Right Thumb ← Thumb Right
        23: 12,  # Left Hip ← Hip Left
        24: 16,  # Right Hip ← Hip Right
        25: 13,  # Left Knee ← Knee Left
        26: 17,  # Right Knee ← Knee Right
        27: 14,  # Left Ankle ← Ankle Left
        28: 18,  # Right Ankle ← Ankle Right
        31: 15,  # Left Foot Index ← Foot Left
        32: 19,  # Right Foot Index ← Foot Right
    }

    for mp_idx, ntu_idx in mapping.items():
        mp_skeleton[:, mp_idx, :] = ntu_skeleton[:, ntu_idx, :]

    # 近似缺失的眼睛、耳朵、嘴巴位置（從頭部位置）
    head_pos = ntu_skeleton[:, 3, :]  # Head
    mp_skeleton[:, 1:7, :] = head_pos[:, np.newaxis, :]  # Eyes
    mp_skeleton[:, 7:9, :] = head_pos[:, np.newaxis, :]  # Ears
    mp_skeleton[:, 9:11, :] = head_pos[:, np.newaxis, :]  # Mouth

    # 近似小指位置（從手部位置）
    mp_skeleton[:, 17, :] = ntu_skeleton[:, 7, :]   # Left Pinky ≈ Hand Left
    mp_skeleton[:, 18, :] = ntu_skeleton[:, 11, :]  # Right Pinky ≈ Hand Right

    # 近似腳跟位置（從腳踝位置）
    mp_skeleton[:, 29, :] = ntu_skeleton[:, 14, :]  # Left Heel ≈ Ankle Left
    mp_skeleton[:, 30, :] = ntu_skeleton[:, 18, :]  # Right Heel ≈ Ankle Right

    return mp_skeleton


# =============================================================================
# 測試
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing MediaPipe 33-Point One-Shot Action Recognition Model")
    print("=" * 70)

    # 建立模型
    model = OneShotActionRecognitionMediaPipe(in_channels=3, base_channels=64)
    model.eval()

    # 顯示模型結構資訊
    print("\n[Model Architecture]")
    print(f"  - Input joints: 33 (MediaPipe Pose)")
    print(f"  - Scale 1 (joints): 33 nodes")
    print(f"  - Scale 2 (parts): 12 nodes")
    print(f"  - Scale 3 (super-parts): 6 nodes")

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")

    # 建立假資料
    batch_size = 1
    seq_length = 64
    num_joints = 33
    channels = 3

    query = torch.randn(batch_size, seq_length, num_joints, channels)
    support = torch.randn(batch_size, seq_length, num_joints, channels)

    print("\n[Input Shape]")
    print(f"  - Query: {query.shape}")
    print(f"  - Support: {support.shape}")

    # 測試特徵提取
    print("\n[Feature Extraction]")
    features = model.extract_features(query)
    print(f"  - Scale 1 features: {features[0].shape}")
    print(f"  - Scale 2 features: {features[1].shape}")
    print(f"  - Scale 3 features: {features[2].shape}")

    # 測試相似度計算
    print("\n[Similarity Computation]")
    with torch.no_grad():
        score = model.compute_similarity(query, support)
        print(f"  - Similarity score: {score.item():.4f}")

    # 測試完整推論
    print("\n[Full Inference]")
    support_set = [
        (torch.randn(1, 64, 33, 3), "waving"),
        (torch.randn(1, 64, 33, 3), "falling"),
        (torch.randn(1, 64, 33, 3), "walking"),
    ]

    predicted, all_scores = model(query, support_set)
    print(f"  - Predicted action: {predicted}")
    print("  - All scores:")
    for score, label in all_scores:
        print(f"      {label}: {score:.4f}")

    print("\n" + "=" * 70)
    print("Model test completed successfully!")
    print("=" * 70)
