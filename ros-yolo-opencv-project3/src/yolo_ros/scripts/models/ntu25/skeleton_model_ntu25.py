#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NTU RGB+D 25 點 One-Shot 動作辨識模型

實作論文「One-Shot Action Recognition via Multi-Scale Spatial-Temporal
Skeleton Matching」，使用原生 NTU RGB+D / Kinect v2 的 25 關節格式。

此模型可直接用於：
- NTU RGB+D 訓練資料集
- Azure Kinect DK 實時推論（32點轉25點）

主要元件：
- 自適應圖卷積網路 (AGCN)
- Earth Mover's Distance (EMD) 最佳匹配
- 多尺度時空匹配

關節定義（NTU RGB+D / Kinect v2）：
 0: Spine Base      1: Spine Mid       2: Neck            3: Head
 4: Shoulder Left   5: Elbow Left      6: Wrist Left      7: Hand Left
 8: Shoulder Right  9: Elbow Right    10: Wrist Right    11: Hand Right
12: Hip Left       13: Knee Left      14: Ankle Left     15: Foot Left
16: Hip Right      17: Knee Right     18: Ankle Right    19: Foot Right
20: Spine          21: Hand Tip Left  22: Thumb Left     23: Hand Tip Right
24: Thumb Right
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import IntEnum

try:
    import ot  # Python Optimal Transport 函式庫
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("警告：找不到 'pot' 函式庫，將使用簡化版 EMD。")


# =============================================================================
# NTU RGB+D 25 關節定義
# =============================================================================

class NTUJoint(IntEnum):
    """NTU RGB+D (Kinect v2) 25 個關節索引"""
    SPINE_BASE = 0
    SPINE_MID = 1
    NECK = 2
    HEAD = 3
    SHOULDER_LEFT = 4
    ELBOW_LEFT = 5
    WRIST_LEFT = 6
    HAND_LEFT = 7
    SHOULDER_RIGHT = 8
    ELBOW_RIGHT = 9
    WRIST_RIGHT = 10
    HAND_RIGHT = 11
    HIP_LEFT = 12
    KNEE_LEFT = 13
    ANKLE_LEFT = 14
    FOOT_LEFT = 15
    HIP_RIGHT = 16
    KNEE_RIGHT = 17
    ANKLE_RIGHT = 18
    FOOT_RIGHT = 19
    SPINE = 20
    HAND_TIP_LEFT = 21
    THUMB_LEFT = 22
    HAND_TIP_RIGHT = 23
    THUMB_RIGHT = 24


class NTUGraph:
    """
    NTU RGB+D 25 關節的圖結構

    基於人體骨架的自然連接定義邊，並根據論文設計多尺度空間池化群組。
    """

    def __init__(self):
        self.num_joints = 25

        # 關節名稱
        self.joint_names = [
            'spine_base', 'spine_mid', 'neck', 'head',
            'shoulder_left', 'elbow_left', 'wrist_left', 'hand_left',
            'shoulder_right', 'elbow_right', 'wrist_right', 'hand_right',
            'hip_left', 'knee_left', 'ankle_left', 'foot_left',
            'hip_right', 'knee_right', 'ankle_right', 'foot_right',
            'spine', 'hand_tip_left', 'thumb_left', 'hand_tip_right', 'thumb_right'
        ]

        # 定義邊（骨架連接）- 基於 Kinect v2 骨架結構
        self.edges = [
            # 脊椎
            (0, 1), (1, 20), (20, 2), (2, 3),
            # 左臂
            (20, 4), (4, 5), (5, 6), (6, 7),
            (7, 21), (7, 22),  # 手指
            # 右臂
            (20, 8), (8, 9), (9, 10), (10, 11),
            (11, 23), (11, 24),  # 手指
            # 左腿
            (0, 12), (12, 13), (13, 14), (14, 15),
            # 右腿
            (0, 16), (16, 17), (17, 18), (18, 19),
        ]

        # 中心關節（脊椎中點，用於空間配置）
        self.center = NTUJoint.SPINE  # 20

        # =====================================================================
        # 尺度 2 的空間池化群組（10 部位）
        # 將具有相似語義的關節分組
        # =====================================================================
        self.scale2_groups = {
            0: [3],                           # 頭部
            1: [0, 1, 2, 20],                 # 軀幹（脊椎）
            2: [4, 5],                        # 左上臂
            3: [6, 7, 21, 22],                # 左手（手腕+手+手指）
            4: [8, 9],                        # 右上臂
            5: [10, 11, 23, 24],              # 右手（手腕+手+手指）
            6: [12, 13],                      # 左大腿
            7: [14, 15],                      # 左小腿+腳
            8: [16, 17],                      # 右大腿
            9: [18, 19],                      # 右小腿+腳
        }

        # =====================================================================
        # 尺度 3 的空間池化群組（5 超級部位）
        # =====================================================================
        self.scale3_groups = {
            0: [2, 3],                                    # 頭頸
            1: [0, 1, 20],                                # 軀幹
            2: [4, 5, 6, 7, 21, 22],                      # 左臂（含手指）
            3: [8, 9, 10, 11, 23, 24],                    # 右臂（含手指）
            4: [12, 13, 14, 15, 16, 17, 18, 19],          # 雙腿
        }

    def get_adjacency(self):
        """取得圖的鄰接矩陣"""
        A = np.zeros((self.num_joints, self.num_joints), dtype=np.float32)

        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1

        # 加入自環
        A = A + np.eye(self.num_joints, dtype=np.float32)

        return A

    def get_scale2_adjacency(self):
        """取得尺度 2（10 部位）的鄰接矩陣"""
        num_parts = len(self.scale2_groups)
        A = np.zeros((num_parts, num_parts), dtype=np.float32)

        # 定義部位間的連接
        edges = [
            (0, 1),      # 頭部 - 軀幹
            (1, 2),      # 軀幹 - 左上臂
            (1, 4),      # 軀幹 - 右上臂
            (2, 3),      # 左上臂 - 左手
            (4, 5),      # 右上臂 - 右手
            (1, 6),      # 軀幹 - 左大腿
            (1, 8),      # 軀幹 - 右大腿
            (6, 7),      # 左大腿 - 左小腿
            (8, 9),      # 右大腿 - 右小腿
        ]

        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1

        A = A + np.eye(num_parts, dtype=np.float32)
        return A

    def get_scale3_adjacency(self):
        """取得尺度 3（5 超級部位）的鄰接矩陣"""
        num_parts = len(self.scale3_groups)
        A = np.zeros((num_parts, num_parts), dtype=np.float32)

        # 定義連接
        edges = [
            (0, 1),  # 頭頸 - 軀幹
            (1, 2),  # 軀幹 - 左臂
            (1, 3),  # 軀幹 - 右臂
            (1, 4),  # 軀幹 - 雙腿
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
    """基本圖卷積層"""

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
        x = torch.einsum('nctv,vw->nctw', x, A)

        # 通道混合
        x = self.conv(x)
        x = self.bn(x)

        return x


class TemporalConv(nn.Module):
    """時間卷積層"""

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
    """自適應圖卷積區塊"""

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

class NTUSkeletonEmbedding(nn.Module):
    """
    使用 AGCN 的多尺度骨架嵌入網路（NTU 25 點版本）

    架構：
    - 6 個共享 AGC 區塊
    - 每個空間尺度各有 3 個獨立區塊
    """

    def __init__(self, in_channels=3, base_channels=64, num_classes=None):
        super().__init__()

        self.graph = NTUGraph()

        # 取得鄰接矩陣
        A1 = self.graph.get_adjacency()
        A2 = self.graph.get_scale2_adjacency()
        A3 = self.graph.get_scale3_adjacency()

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

        # 尺度 1 專用區塊（25 關節）
        self.scale1_blocks = nn.ModuleList([
            AGCBlock(base_channels * 2, base_channels * 4, A1, stride=2),
            AGCBlock(base_channels * 4, base_channels * 4, A1),
            AGCBlock(base_channels * 4, base_channels * 4, A1),
        ])

        # 尺度 2 專用區塊（10 部位）
        self.scale2_blocks = nn.ModuleList([
            AGCBlock(base_channels * 2, base_channels * 4, A2, stride=2),
            AGCBlock(base_channels * 4, base_channels * 4, A2),
            AGCBlock(base_channels * 4, base_channels * 4, A2),
        ])

        # 尺度 3 專用區塊（5 超級部位）
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
        根據空間群組池化特徵

        參數：
            x: (N, C, T, V)
            groups: 將 group_id 映射到關節索引列表的字典

        回傳：
            池化後的特徵 (N, C, T, num_groups)
        """
        N, C, T, V = x.shape
        num_groups = len(groups)
        pooled = torch.zeros(N, C, T, num_groups, device=x.device, dtype=x.dtype)

        for g_id, joints in groups.items():
            pooled[:, :, :, g_id] = x[:, :, :, joints].mean(dim=-1)

        return pooled

    def forward(self, x, return_multi_scale=False):
        """
        前向傳播

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

        # 尺度 1（25 關節）
        x1 = x
        for block in self.scale1_blocks:
            x1 = block(x1)

        # 尺度 2（10 部位）- 先池化再處理
        x2 = self.spatial_pool(x, self.graph.scale2_groups)
        for block in self.scale2_blocks:
            x2 = block(x2)

        # 尺度 3（5 超級部位）
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
    """基於 Earth Mover's Distance 的最佳匹配"""

    def __init__(self, use_pot=True):
        self.use_pot = use_pot and HAS_POT

    def compute_distance_matrix(self, X, Y):
        """計算成對餘弦距離矩陣"""
        X_norm = F.normalize(X, p=2, dim=0)
        Y_norm = F.normalize(Y, p=2, dim=0)
        similarity = torch.mm(X_norm.t(), Y_norm)
        distance = 1 - similarity
        return distance

    def compute_weights(self, X, Y):
        """使用交叉參考機制計算節點權重"""
        Y_mean = Y.mean(dim=1, keepdim=True)
        X_mean = X.mean(dim=1, keepdim=True)

        r = torch.mm(X.t(), Y_mean).squeeze()
        c = torch.mm(Y.t(), X_mean).squeeze()

        r = F.relu(r) + 1e-6
        c = F.relu(c) + 1e-6

        r = r / r.sum()
        c = c / c.sum()

        return r, c

    def compute_emd(self, X, Y):
        """計算兩個特徵集之間的 EMD"""
        if X.dim() == 4:
            N, C, T, V = X.shape
            X = X.view(C, -1)
        if Y.dim() == 4:
            Y = Y.view(Y.shape[1], -1)

        D = self.compute_distance_matrix(X, Y)
        r, c = self.compute_weights(X, Y)

        if self.use_pot:
            r_np = r.detach().cpu().numpy().astype(np.float64)
            c_np = c.detach().cpu().numpy().astype(np.float64)
            D_np = D.detach().cpu().numpy().astype(np.float64)

            pi = ot.emd(r_np, c_np, D_np)
            pi = torch.from_numpy(pi).to(X.device).float()
        else:
            pi = self.sinkhorn(r, c, D)

        similarity = 1 - D
        score = (similarity * pi).sum()
        emd = (D * pi).sum()

        return emd, score

    def sinkhorn(self, r, c, D, reg=0.1, max_iter=100):
        """Sinkhorn-Knopp 演算法"""
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
    """用於 one-shot 動作辨識的多尺度與跨尺度匹配"""

    def __init__(self):
        super().__init__()
        self.emd = EMDMatcher()

    def temporal_pool(self, x, scale):
        """池化時間維度以建立較粗的尺度"""
        if scale == 1:
            return x
        return F.avg_pool2d(x, kernel_size=(scale, 1), stride=(scale, 1))

    def compute_multi_scale_score(self, X_scales, Y_scales):
        """計算多尺度匹配分數"""
        total_score = 0
        for x, y in zip(X_scales, Y_scales):
            _, score = self.emd.compute_emd(x, y)
            total_score += score
        return total_score

    def compute_multi_temporal_score(self, X, Y):
        """計算多時間尺度匹配分數"""
        total_score = 0
        for scale in [1, 2, 4]:
            x_t = self.temporal_pool(X, scale)
            y_t = self.temporal_pool(Y, scale)
            _, score = self.emd.compute_emd(x_t, y_t)
            total_score += score
        return total_score

    def compute_cross_scale_score(self, X_scales, Y_scales):
        """計算跨尺度匹配分數"""
        total_score = 0
        for i in range(len(X_scales)):
            for j in range(len(Y_scales)):
                if i != j:
                    x = F.adaptive_avg_pool2d(X_scales[i], (1, 1))
                    y = F.adaptive_avg_pool2d(Y_scales[j], (1, 1))
                    x = x.view(x.shape[0], x.shape[1], -1)
                    y = y.view(y.shape[0], y.shape[1], -1)
                    _, score = self.emd.compute_emd(x, y)
                    total_score += score
        return total_score

    def forward(self, X_features, Y_features):
        """計算查詢和支持之間的總匹配分數"""
        ms_score = self.compute_multi_scale_score(X_features, Y_features)
        mt_score = self.compute_multi_temporal_score(X_features[0], Y_features[0])
        cs_score = self.compute_cross_scale_score(X_features, Y_features)
        total_score = ms_score + mt_score + cs_score
        return total_score


# =============================================================================
# One-Shot 動作辨識模型
# =============================================================================

class OneShotActionRecognitionNTU25(nn.Module):
    """
    完整的 one-shot 動作辨識模型（NTU 25 點版本）

    可用於：
    - NTU RGB+D 訓練資料
    - Azure Kinect DK 實時推論
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.embedding = NTUSkeletonEmbedding(
            in_channels=in_channels,
            base_channels=base_channels
        )
        self.matcher = MultiScaleMatcher()
        self.num_joints = 25

    def extract_features(self, x):
        """
        從骨架序列中提取多尺度特徵

        參數：
            x: (N, T, V, C) 骨架序列

        回傳：
            不同尺度的特徵元組
        """
        return self.embedding(x, return_multi_scale=True)

    def compute_similarity(self, query, support):
        """
        計算查詢和支持序列之間的相似度

        參數：
            query: (1, T, V, C) 查詢骨架序列
            support: (1, T, V, C) 支持骨架序列

        回傳：
            相似度分數
        """
        query_features = self.extract_features(query)
        support_features = self.extract_features(support)
        score = self.matcher(query_features, support_features)
        return score

    def forward(self, query, support_set):
        """
        執行 one-shot 動作辨識

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

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1], scores


# =============================================================================
# 工具函數
# =============================================================================

def preprocess_ntu_skeleton(keypoints, target_length=64):
    """
    預處理 NTU 25 點骨架序列作為模型輸入

    參數：
        keypoints: (T, 25, C) 原始關鍵點
        target_length: 目標序列長度

    回傳：
        預處理後的張量 (1, T, 25, C)
    """
    T, N, C = keypoints.shape

    # 時間插值/取樣
    if T != target_length:
        indices = np.linspace(0, T - 1, target_length).astype(int)
        keypoints = keypoints[indices]

    # 正規化座標（以 Spine Base 為中心）
    center = keypoints[:, NTUJoint.SPINE_BASE, :3]  # 使用 spine_base 作為中心
    keypoints[:, :, :3] -= center[:, np.newaxis, :]

    # 轉換為張量
    tensor = torch.from_numpy(keypoints).float().unsqueeze(0)

    return tensor


def load_ntu_skeleton_file(filepath):
    """
    載入 NTU RGB+D 骨架檔案

    參數：
        filepath: .skeleton 檔案路徑

    回傳：
        骨架序列 (T, 25, 3) 和 metadata
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_frames = int(lines[0].strip())
    skeletons = []
    line_idx = 1

    for frame_idx in range(num_frames):
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1

        if num_bodies == 0:
            # 沒有人，填入零
            skeletons.append(np.zeros((25, 3), dtype=np.float32))
            continue

        # 讀取第一個人的骨架
        # 跳過 body info (10 個數值)
        line_idx += 1

        num_joints = int(lines[line_idx].strip())
        line_idx += 1

        joints = []
        for j in range(num_joints):
            joint_data = list(map(float, lines[line_idx].strip().split()))
            # 取前 3 個值 (x, y, z)
            joints.append(joint_data[:3])
            line_idx += 1

        skeletons.append(np.array(joints, dtype=np.float32))

        # 跳過其他人
        for _ in range(num_bodies - 1):
            line_idx += 1  # body info
            num_joints_other = int(lines[line_idx].strip())
            line_idx += 1
            line_idx += num_joints_other

    skeletons = np.array(skeletons, dtype=np.float32)
    return skeletons


if __name__ == '__main__':
    # 測試模型
    print("=" * 60)
    print("NTU 25 點 One-Shot 動作辨識模型測試")
    print("=" * 60)

    # 建立模型
    model = OneShotActionRecognitionNTU25(in_channels=3, base_channels=64)
    model.eval()

    # 列印模型資訊
    graph = NTUGraph()
    print(f"\n關節數量: {graph.num_joints}")
    print(f"尺度 2 部位數: {len(graph.scale2_groups)}")
    print(f"尺度 3 超級部位數: {len(graph.scale3_groups)}")

    # 建立假資料
    batch_size = 1
    seq_length = 64
    num_joints = 25
    channels = 3

    query = torch.randn(batch_size, seq_length, num_joints, channels)
    support = torch.randn(batch_size, seq_length, num_joints, channels)

    # 測試特徵提取
    print("\n特徵提取測試:")
    features = model.extract_features(query)
    print(f"  尺度 1 (25 關節): {features[0].shape}")
    print(f"  尺度 2 (10 部位): {features[1].shape}")
    print(f"  尺度 3 (5 超級部位): {features[2].shape}")

    # 測試相似度計算
    print("\n相似度計算測試:")
    with torch.no_grad():
        score = model.compute_similarity(query, support)
        print(f"  相似度分數: {score.item():.4f}")

    # 測試完整推論
    print("\n完整推論測試:")
    support_set = [
        (torch.randn(1, 64, 25, 3), "drink_water"),
        (torch.randn(1, 64, 25, 3), "eat_meal"),
        (torch.randn(1, 64, 25, 3), "brush_teeth"),
        (torch.randn(1, 64, 25, 3), "brush_hair"),
        (torch.randn(1, 64, 25, 3), "drop"),
    ]

    predicted, all_scores = model(query, support_set)
    print(f"  預測動作: {predicted}")
    print("  所有分數:")
    for score, label in all_scores:
        print(f"    {label}: {score:.4f}")

    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型參數:")
    print(f"  總參數量: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("模型測試完成！")
    print("=" * 60)
