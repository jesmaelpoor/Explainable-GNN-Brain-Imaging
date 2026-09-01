# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:33:37 2026

@author: JEsmaelpoor
"""

import copy
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, friedmanchisquare, wilcoxon, rankdata, sem
from matplotlib.patches import Patch

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


# =========================================================
# Config
# =========================================================
subsID_1 = [1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 41, 43, 44]
subsID_1 = [1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 41, 43, 44]


data_folder = "C:/Melbourne University/PhD studies_works/Study Five-GNN/NHMRC Project/final codes/sess1_graphs/"

ROI_NAMES = ["IFG", "STGL", "AGL", "STGR", "AGR", "OCC"]

# Feature layout in x:
# [roi(6), audio_avg(20), audio_dyn(2), visual_avg(20), visual_dyn(2)]
BLOCKS = {
    "ROI_code": slice(0, 6),
    "Audio_avg": slice(6, 26),
    "Audio_dyn": slice(26, 28),
    "Visual_avg": slice(28, 48),
    "Visual_dyn": slice(48, 50),
}
FEATURE_BLOCK_NAMES = list(BLOCKS.keys()) + ["Implant_side"]

BLOCK_SIZES = {
    "ROI_code": 6,
    "Audio_avg": 20,
    "Audio_dyn": 2,
    "Visual_avg": 20,
    "Visual_dyn": 2,
    "Implant_side": 1,
}


# =========================================================
# Individual feature labels / groups / colors
# =========================================================
def build_individual_feature_info():
    feature_labels = []
    feature_groups = []

    for r in ROI_NAMES:
        feature_labels.append(f"ROI_{r}")
        feature_groups.append("ROI_code")

    for i in range(20):
        feature_labels.append(f"A_avg_{i+1}")
        feature_groups.append("Audio_avg")

    for i in range(2):
        feature_labels.append(f"A_dyn_{i+1}")
        feature_groups.append("Audio_dyn")

    for i in range(20):
        feature_labels.append(f"V_avg_{i+1}")
        feature_groups.append("Visual_avg")

    for i in range(2):
        feature_labels.append(f"V_dyn_{i+1}")
        feature_groups.append("Visual_dyn")

    feature_labels.append("Implant_side")
    feature_groups.append("Implant_side")

    return feature_labels, feature_groups


INDIV_FEATURE_NAMES, INDIV_FEATURE_GROUPS = build_individual_feature_info()

GROUP_COLORS = {
    "ROI_code": "#7f7f7f",
    "Audio_avg": "#1f77b4",
    "Audio_dyn": "#17becf",
    "Visual_avg": "#d62728",
    "Visual_dyn": "#ff9896",
    "Implant_side": "#9467bd",
}


# =========================================================
# Model
# =========================================================
class GATWithGlobalPooling(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=1, heads=1, attention_dropout=0.1):
        super(GATWithGlobalPooling, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim1, heads=heads, dropout=attention_dropout, concat=True)
        self.conv2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=1, dropout=attention_dropout, concat=False)

        self.fc = torch.nn.Linear(hidden_dim2 + 1, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch, side):
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        side = side.view(-1, 1).float()
        x = torch.cat([x, side], dim=1)

        x = self.fc(x)
        x = self.sigmoid(x)

        return x


# =========================================================
# Data utils
# =========================================================
def load_all_graphs(folder, subject_list):
    all_graphs = []
    for subject in subject_list:
        for file in os.listdir(folder):
            if file.startswith(f"sub_{subject}_") and file.endswith(".pt"):
                graph = torch.load(os.path.join(folder, file), weights_only=False)
                all_graphs.append(graph)
    return all_graphs


# =========================================================
# Train / test
# =========================================================
def train_gat(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        out = model(batch_data.x.float(), batch_data.edge_index, batch_data.batch, batch_data.side)
        target = batch_data.y[:, 1].float().unsqueeze(1)

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def test_gat(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch_data in loader:
        batch_data = batch_data.to(device)

        out = model(batch_data.x.float(), batch_data.edge_index, batch_data.batch, batch_data.side)
        target = batch_data.y[:, 1].float().unsqueeze(1)

        loss = criterion(out, target)
        total_loss += loss.item()

        all_predictions.append(out.cpu())
        all_targets.append(target.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_predicted_score = all_predictions.mean().item()
    avg_true_score = all_targets.mean().item()

    return total_loss / len(loader), avg_predicted_score, avg_true_score


# =========================================================
# Interpretation helpers
# =========================================================
def has_all_6_agg_nodes(graph):
    roi_code = graph.x[:, 0:6]
    roi_sum = roi_code.abs().sum(dim=1)
    agg_nodes = torch.where(roi_sum < 1e-8)[0]
    return len(agg_nodes) == 6


def compute_train_feature_means(train_graphs):
    x_all = torch.cat([g.x.float() for g in train_graphs], dim=0)
    side_all = torch.cat([g.side.view(-1).float() for g in train_graphs], dim=0)

    mean_x = x_all.mean(dim=0)
    mean_side = side_all.mean().item()

    return mean_x, mean_side


@torch.no_grad()
def predict_single_graph(model, graph, device):
    model.eval()
    g = copy.deepcopy(graph).to(device)
    batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
    out = model(g.x.float(), g.edge_index, batch, g.side)
    return out.item()


def clone_graph(graph):
    return copy.deepcopy(graph)


def mask_feature_block(graph, block_slice, mean_x):
    g = clone_graph(graph)
    g.x[:, block_slice] = mean_x[block_slice].view(1, -1).repeat(g.x.size(0), 1)
    return g


def mask_single_feature(graph, feature_idx, mean_x):
    g = clone_graph(graph)
    g.x[:, feature_idx] = mean_x[feature_idx]
    return g


def mask_side(graph, mean_side):
    g = clone_graph(graph)
    g.side = torch.tensor([mean_side], dtype=g.side.dtype)
    return g


def mask_nodes_with_mean(graph, node_idx, mean_x):
    g = clone_graph(graph)
    if len(node_idx) > 0:
        g.x[node_idx, :] = mean_x.view(1, -1).repeat(len(node_idx), 1)
    return g


def remove_edges_by_mask(graph, remove_mask):
    g = clone_graph(graph)

    keep_mask = ~remove_mask
    g.edge_index = g.edge_index[:, keep_mask]

    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        g.edge_attr = g.edge_attr[keep_mask]

    if hasattr(g, "edge_weight") and g.edge_weight is not None:
        g.edge_weight = g.edge_weight[keep_mask]

    return g


def get_graph_roi_info(graph):
    roi_code = graph.x[:, 0:6]
    roi_sum = roi_code.abs().sum(dim=1)

    agg_nodes = torch.where(roi_sum < 1e-8)[0]
    agg_nodes = torch.sort(agg_nodes).values

    roi_nodes = []
    for r in range(6):
        idx = torch.where(roi_code[:, r] > 0.5)[0]
        roi_nodes.append(idx)

    if len(agg_nodes) != 6:
        print(f"Warning: expected 6 aggregation nodes, found {len(agg_nodes)}")

    return roi_nodes, agg_nodes


def get_edge_group_masks(graph):
    roi_nodes, agg_nodes = get_graph_roi_info(graph)

    src = graph.edge_index[0]
    dst = graph.edge_index[1]

    within_roi_masks = []
    roi_to_agg_masks = []

    for r in range(6):
        rn = roi_nodes[r]

        if len(rn) == 0:
            within_roi_masks.append(torch.zeros(src.size(0), dtype=torch.bool))
            roi_to_agg_masks.append(torch.zeros(src.size(0), dtype=torch.bool))
            continue

        agg = agg_nodes[r]

        within_mask = torch.isin(src, rn) & torch.isin(dst, rn)

        roi_agg_mask = ((torch.isin(src, rn) & (dst == agg)) |
                        (torch.isin(dst, rn) & (src == agg)))

        within_roi_masks.append(within_mask)
        roi_to_agg_masks.append(roi_agg_mask)

    inter_agg_masks = {}
    for i in range(6):
        for j in range(i + 1, 6):
            ai = agg_nodes[i]
            aj = agg_nodes[j]

            mask = (((src == ai) & (dst == aj)) |
                    ((src == aj) & (dst == ai)))
            inter_agg_masks[(i, j)] = mask

    return within_roi_masks, roi_to_agg_masks, inter_agg_masks


def nanmean_safe(arr, axis=0, fallback_shape=None):
    arr = np.array(arr, dtype=float)

    if arr.size == 0:
        if fallback_shape is None:
            return np.nan
        return np.full(fallback_shape, np.nan, dtype=float)

    return np.nanmean(arr, axis=axis)


# =========================================================
# Statistics helpers
# =========================================================
def benjamini_hochberg(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def kendalls_w(matrix):
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape

    ranks = np.apply_along_axis(rankdata, 1, matrix)
    Rj = np.sum(ranks, axis=0)
    Rbar = np.mean(Rj)
    S = np.sum((Rj - Rbar) ** 2)

    W = 12 * S / (n ** 2 * (k ** 3 - k))
    return W


def run_repeated_stats(data_mat, labels, analysis_name):
    data_mat = np.asarray(data_mat, dtype=float)

    valid_mask = ~np.isnan(data_mat).any(axis=1)
    data_clean = data_mat[valid_mask, :]

    stats_summary = []
    pairwise_rows = []

    if data_clean.shape[0] < 2:
        stats_summary.append({
            "analysis": analysis_name,
            "n_subjects": data_clean.shape[0],
            "friedman_chi2": np.nan,
            "friedman_p": np.nan,
            "kendalls_W": np.nan
        })
        return pd.DataFrame(stats_summary), pd.DataFrame(pairwise_rows)

    friedman_stat, friedman_p = friedmanchisquare(*[data_clean[:, i] for i in range(data_clean.shape[1])])
    W = kendalls_w(data_clean)

    stats_summary.append({
        "analysis": analysis_name,
        "n_subjects": data_clean.shape[0],
        "friedman_chi2": friedman_stat,
        "friedman_p": friedman_p,
        "kendalls_W": W
    })

    pairs = list(itertools.combinations(range(len(labels)), 2))
    raw_pvals = []
    wilcoxon_stats = []

    for i, j in pairs:
        try:
            stat, p = wilcoxon(data_clean[:, i], data_clean[:, j], zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        wilcoxon_stats.append(stat)
        raw_pvals.append(p)

    adj_pvals = benjamini_hochberg([1.0 if np.isnan(p) else p for p in raw_pvals])

    for idx, (i, j) in enumerate(pairs):
        pairwise_rows.append({
            "analysis": analysis_name,
            "group_1": labels[i],
            "group_2": labels[j],
            "wilcoxon_stat": wilcoxon_stats[idx],
            "raw_p": raw_pvals[idx],
            "fdr_p": adj_pvals[idx]
        })

    return pd.DataFrame(stats_summary), pd.DataFrame(pairwise_rows)


# =========================================================
# Plot helpers
# =========================================================
def plot_heatmap(mat, labels, title, cmap="viridis", colorbar_label="Mean absolute prediction change (percentage points)"):
    plt.figure(figsize=(7, 6))
    plt.imshow(mat, cmap=cmap)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_individual_feature_importance_with_sem(values_mean, values_sem, feature_names, feature_groups, title,
                                                ylabel="Mean absolute prediction change (percentage points)"):
    x = np.arange(len(values_mean))
    colors = [GROUP_COLORS[g] for g in feature_groups]

    plt.figure(figsize=(18, 6))
    plt.bar(x, values_mean, color=colors, alpha=0.9)

    plt.errorbar(
        x, values_mean, yerr=values_sem,
        fmt='none', ecolor='black', elinewidth=1, capsize=2, alpha=0.8
    )

    tick_positions = []
    tick_labels = []
    for i, name in enumerate(feature_names):
        if feature_groups[i] in ["ROI_code", "Audio_dyn", "Visual_dyn", "Implant_side"]:
            tick_positions.append(i)
            tick_labels.append(name)
        elif name.endswith("_1") or name.endswith("_5") or name.endswith("_10") or name.endswith("_15") or name.endswith("_20"):
            tick_positions.append(i)
            tick_labels.append(name)

    plt.xticks(tick_positions, tick_labels, rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)

    group_boundaries = [6, 26, 28, 48, 50]
    for b in group_boundaries:
        plt.axvline(b - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    legend_elements = [
        Patch(facecolor=GROUP_COLORS["ROI_code"], label="ROI_code"),
        Patch(facecolor=GROUP_COLORS["Audio_avg"], label="Audio_avg"),
        Patch(facecolor=GROUP_COLORS["Audio_dyn"], label="Audio_dyn"),
        Patch(facecolor=GROUP_COLORS["Visual_avg"], label="Visual_avg"),
        Patch(facecolor=GROUP_COLORS["Visual_dyn"], label="Visual_dyn"),
        Patch(facecolor=GROUP_COLORS["Implant_side"], label="Implant_side"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.show()


def plot_boxplot_with_jitter(data_mat, labels, title, ylabel="Mean absolute prediction change (percentage points)"):
    data_mat = np.asarray(data_mat, dtype=float)

    valid_mask = ~np.isnan(data_mat).any(axis=1)
    data_clean = data_mat[valid_mask, :]

    plt.figure(figsize=(9, 6))
    plt.boxplot([data_clean[:, i] for i in range(data_clean.shape[1])],
                labels=labels, showfliers=False)

    rng = np.random.default_rng(42)
    for i in range(data_clean.shape[1]):
        x = np.full(data_clean.shape[0], i + 1, dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=data_clean.shape[0])
        plt.scatter(x + jitter, data_clean[:, i], alpha=0.7, s=25)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# Main LOSO + interpretation
# =========================================================
results = {}

interp_feature_blocks = {}
interp_individual_features = {}

# total
interp_roi_nodes = {}
interp_within_roi_edges = {}
interp_roi_to_agg_edges = {}

# normalized
interp_roi_nodes_norm = {}
interp_within_roi_edges_norm = {}
interp_roi_to_agg_edges_norm = {}

# unchanged
interp_agg_nodes = {}
interp_inter_agg_edges = {}

for test_subject in subsID_1:
    print(f"\nTesting on subject {test_subject}")

    test_graphs = load_all_graphs(data_folder, [test_subject])
    train_subjects = [s for s in subsID_1 if s != test_subject]
    train_graphs = load_all_graphs(data_folder, train_subjects)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    input_dim = train_graphs[0].x.shape[1]
    output_dim = 1
    hidden_dim1 = 24
    hidden_dim2 = 24

    model = GATWithGlobalPooling(
        input_dim,
        hidden_dim1,
        hidden_dim2,
        output_dim,
        heads=1,
        attention_dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = torch.nn.HuberLoss(delta=1.0)

    epochs = 101
    best_train_loss = float("inf")
    patience = 15
    min_improvement = 8e-5
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_gat(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        if best_train_loss - train_loss > min_improvement:
            best_train_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    test_loss, avg_predicted_score, avg_true_score = test_gat(model, test_loader, criterion, device)

    results[test_subject] = {
        "MSE": test_loss,
        "Predicted_BKB": avg_predicted_score,
        "True_BKB": avg_true_score
    }

    print(f"Subject {test_subject}: Test Loss = {test_loss:.4f}")
    print(f"Predicted BKB = {avg_predicted_score:.2f}, True BKB = {avg_true_score:.2f}")

    mean_x, mean_side = compute_train_feature_means(train_graphs)

    feature_block_graph_scores = []
    individual_feature_graph_scores = []

    roi_node_graph_scores = []
    roi_node_graph_scores_norm = []

    agg_node_graph_scores = []

    within_roi_edge_graph_scores = []
    within_roi_edge_graph_scores_norm = []

    roi_to_agg_edge_graph_scores = []
    roi_to_agg_edge_graph_scores_norm = []

    inter_agg_edge_graph_scores = []

    test_graphs_interp = [g for g in test_graphs if has_all_6_agg_nodes(g)]

    n_skipped = len(test_graphs) - len(test_graphs_interp)
    if n_skipped > 0:
        print(f"Skipped {n_skipped} test graphs for subject {test_subject} because they do not contain all 6 aggregation nodes.")

    for g in test_graphs_interp:
        base_pred = predict_single_graph(model, g, device)

        # 1) Feature-block importance
        feat_scores = []
        for block_name in FEATURE_BLOCK_NAMES:
            if block_name == "Implant_side":
                g_mask = mask_side(g, mean_side)
            else:
                g_mask = mask_feature_block(g, BLOCKS[block_name], mean_x)

            pred_mask = predict_single_graph(model, g_mask, device)
            feat_scores.append(abs(base_pred - pred_mask))
        feature_block_graph_scores.append(feat_scores)

        # 1b) Individual feature importance
        indiv_scores = []
        for feat_idx in range(g.x.shape[1]):
            g_mask = mask_single_feature(g, feat_idx, mean_x)
            pred_mask = predict_single_graph(model, g_mask, device)
            indiv_scores.append(abs(base_pred - pred_mask))

        g_mask_side = mask_side(g, mean_side)
        pred_mask_side = predict_single_graph(model, g_mask_side, device)
        indiv_scores.append(abs(base_pred - pred_mask_side))
        individual_feature_graph_scores.append(indiv_scores)

        roi_nodes, agg_nodes = get_graph_roi_info(g)

        # 2) ROI node importance total + normalized
        roi_scores = []
        roi_scores_norm = []
        for r in range(6):
            rn = roi_nodes[r]
            if len(rn) == 0:
                roi_scores.append(np.nan)
                roi_scores_norm.append(np.nan)
            else:
                g_mask = mask_nodes_with_mean(g, rn, mean_x)
                pred_mask = predict_single_graph(model, g_mask, device)
                delta = abs(base_pred - pred_mask)
                roi_scores.append(delta)
                roi_scores_norm.append(delta / len(rn))
        roi_node_graph_scores.append(roi_scores)
        roi_node_graph_scores_norm.append(roi_scores_norm)

        # 3) Aggregation node importance
        agg_scores = []
        for r in range(6):
            if r >= len(agg_nodes):
                agg_scores.append(np.nan)
            else:
                g_mask = mask_nodes_with_mean(g, agg_nodes[r].view(1), mean_x)
                pred_mask = predict_single_graph(model, g_mask, device)
                agg_scores.append(abs(base_pred - pred_mask))
        agg_node_graph_scores.append(agg_scores)

        within_roi_masks, roi_to_agg_masks, inter_agg_masks = get_edge_group_masks(g)

        # 4) Within-ROI edge importance total + normalized
        within_scores = []
        within_scores_norm = []
        for r in range(6):
            mask = within_roi_masks[r]
            n_edges = int(mask.sum().item())
            if n_edges == 0:
                within_scores.append(np.nan)
                within_scores_norm.append(np.nan)
            else:
                g_edge = remove_edges_by_mask(g, mask)
                pred_edge = predict_single_graph(model, g_edge, device)
                delta = abs(base_pred - pred_edge)
                within_scores.append(delta)
                within_scores_norm.append(delta / n_edges)
        within_roi_edge_graph_scores.append(within_scores)
        within_roi_edge_graph_scores_norm.append(within_scores_norm)

        # 5) ROI-to-aggregation edge importance total + normalized
        roiagg_scores = []
        roiagg_scores_norm = []
        for r in range(6):
            mask = roi_to_agg_masks[r]
            n_edges = int(mask.sum().item())
            if n_edges == 0:
                roiagg_scores.append(np.nan)
                roiagg_scores_norm.append(np.nan)
            else:
                g_edge = remove_edges_by_mask(g, mask)
                pred_edge = predict_single_graph(model, g_edge, device)
                delta = abs(base_pred - pred_edge)
                roiagg_scores.append(delta)
                roiagg_scores_norm.append(delta / n_edges)
        roi_to_agg_edge_graph_scores.append(roiagg_scores)
        roi_to_agg_edge_graph_scores_norm.append(roiagg_scores_norm)

        # 6) Inter-aggregation edge importance
        inter_mat = np.full((6, 6), np.nan, dtype=float)
        for (i, j), mask in inter_agg_masks.items():
            if mask.sum().item() == 0:
                continue

            g_edge = remove_edges_by_mask(g, mask)
            pred_edge = predict_single_graph(model, g_edge, device)
            delta = abs(base_pred - pred_edge)

            inter_mat[i, j] = delta
            inter_mat[j, i] = delta

        inter_agg_edge_graph_scores.append(inter_mat)

    interp_feature_blocks[test_subject] = nanmean_safe(feature_block_graph_scores, axis=0, fallback_shape=(6,))
    interp_individual_features[test_subject] = nanmean_safe(individual_feature_graph_scores, axis=0, fallback_shape=(51,))

    interp_roi_nodes[test_subject] = nanmean_safe(roi_node_graph_scores, axis=0, fallback_shape=(6,))
    interp_roi_nodes_norm[test_subject] = nanmean_safe(roi_node_graph_scores_norm, axis=0, fallback_shape=(6,))

    interp_agg_nodes[test_subject] = nanmean_safe(agg_node_graph_scores, axis=0, fallback_shape=(6,))

    interp_within_roi_edges[test_subject] = nanmean_safe(within_roi_edge_graph_scores, axis=0, fallback_shape=(6,))
    interp_within_roi_edges_norm[test_subject] = nanmean_safe(within_roi_edge_graph_scores_norm, axis=0, fallback_shape=(6,))

    interp_roi_to_agg_edges[test_subject] = nanmean_safe(roi_to_agg_edge_graph_scores, axis=0, fallback_shape=(6,))
    interp_roi_to_agg_edges_norm[test_subject] = nanmean_safe(roi_to_agg_edge_graph_scores_norm, axis=0, fallback_shape=(6,))

    interp_inter_agg_edges[test_subject] = nanmean_safe(inter_agg_edge_graph_scores, axis=0, fallback_shape=(6, 6))


# =========================================================
# Save performance results
# =========================================================
df = pd.DataFrame.from_dict(results, orient="index")
df.to_csv("graph_level_results_bkb_sideFedToFC.csv", index_label="Subject_ID")

true_scores = df["True_BKB"].values
pred_scores = df["Predicted_BKB"].values

r, p = pearsonr(true_scores, pred_scores)

plt.figure(figsize=(7, 7))
plt.scatter(true_scores, pred_scores)

m, b = np.polyfit(true_scores, pred_scores, 1)
x_line = np.linspace(true_scores.min(), true_scores.max(), 100)
plt.plot(x_line, m * x_line + b)

plt.xlabel("True BKB")
plt.ylabel("Predicted BKB")
plt.title(f"Predicted vs True BKB\nR = {r:.3f}, p = {p:.4g}")
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# Group-level interpretation results
# =========================================================
feature_block_mat = np.vstack([interp_feature_blocks[s] for s in subsID_1 if s in interp_feature_blocks])
individual_feature_mat = np.vstack([interp_individual_features[s] for s in subsID_1 if s in interp_individual_features])

roi_node_mat = np.vstack([interp_roi_nodes[s] for s in subsID_1 if s in interp_roi_nodes])
roi_node_norm_mat = np.vstack([interp_roi_nodes_norm[s] for s in subsID_1 if s in interp_roi_nodes_norm])

agg_node_mat = np.vstack([interp_agg_nodes[s] for s in subsID_1 if s in interp_agg_nodes])

within_roi_edge_mat = np.vstack([interp_within_roi_edges[s] for s in subsID_1 if s in interp_within_roi_edges])
within_roi_edge_norm_mat = np.vstack([interp_within_roi_edges_norm[s] for s in subsID_1 if s in interp_within_roi_edges_norm])

roi_to_agg_edge_mat = np.vstack([interp_roi_to_agg_edges[s] for s in subsID_1 if s in interp_roi_to_agg_edges])
roi_to_agg_edge_norm_mat = np.vstack([interp_roi_to_agg_edges_norm[s] for s in subsID_1 if s in interp_roi_to_agg_edges_norm])

inter_agg_stack = np.stack([interp_inter_agg_edges[s] for s in subsID_1 if s in interp_inter_agg_edges], axis=0)

# ---------------------------------------------------------
# Convert from 0-1 to percentage points
# ---------------------------------------------------------
feature_block_mat *= 100
individual_feature_mat *= 100

roi_node_mat *= 100
roi_node_norm_mat *= 100

agg_node_mat *= 100

within_roi_edge_mat *= 100
within_roi_edge_norm_mat *= 100

roi_to_agg_edge_mat *= 100
roi_to_agg_edge_norm_mat *= 100

inter_agg_stack *= 100

# ---------------------------------------------------------
# Derived matrices / group summaries AFTER scaling
# ---------------------------------------------------------
feature_block_norm_mat = feature_block_mat.copy()
for i, name in enumerate(FEATURE_BLOCK_NAMES):
    feature_block_norm_mat[:, i] = feature_block_norm_mat[:, i] / BLOCK_SIZES[name]

group_feature_blocks = np.nanmean(feature_block_mat, axis=0)
group_feature_blocks_norm = np.nanmean(feature_block_norm_mat, axis=0)

group_individual_features = np.nanmean(individual_feature_mat, axis=0)
group_individual_features_sem = sem(individual_feature_mat, axis=0, nan_policy='omit')

group_roi_nodes = np.nanmean(roi_node_mat, axis=0)
group_roi_nodes_norm = np.nanmean(roi_node_norm_mat, axis=0)

group_agg_nodes = np.nanmean(agg_node_mat, axis=0)

group_within_roi_edges = np.nanmean(within_roi_edge_mat, axis=0)
group_within_roi_edges_norm = np.nanmean(within_roi_edge_norm_mat, axis=0)

group_roi_to_agg_edges = np.nanmean(roi_to_agg_edge_mat, axis=0)
group_roi_to_agg_edges_norm = np.nanmean(roi_to_agg_edge_norm_mat, axis=0)

group_inter_agg_edges = np.nanmean(inter_agg_stack, axis=0)


# =========================================================
# Statistics
# =========================================================
stats_summary_list = []
pairwise_list = []

df_stats, df_pairs = run_repeated_stats(feature_block_mat, FEATURE_BLOCK_NAMES, "feature_blocks_total")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(feature_block_norm_mat, FEATURE_BLOCK_NAMES, "feature_blocks_normalized")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(roi_node_mat, ROI_NAMES, "roi_nodes_total")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(roi_node_norm_mat, ROI_NAMES, "roi_nodes_normalized")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(agg_node_mat, ROI_NAMES, "aggregation_nodes")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(within_roi_edge_mat, ROI_NAMES, "within_roi_edges_total")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(within_roi_edge_norm_mat, ROI_NAMES, "within_roi_edges_normalized")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(roi_to_agg_edge_mat, ROI_NAMES, "roi_to_agg_edges_total")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

df_stats, df_pairs = run_repeated_stats(roi_to_agg_edge_norm_mat, ROI_NAMES, "roi_to_agg_edges_normalized")
stats_summary_list.append(df_stats)
pairwise_list.append(df_pairs)

stats_summary_df = pd.concat(stats_summary_list, ignore_index=True)
pairwise_df = pd.concat(pairwise_list, ignore_index=True)

stats_summary_df.to_csv("explainability_stats_summary.csv", index=False)
pairwise_df.to_csv("explainability_pairwise_stats.csv", index=False)


# =========================================================
# Save interpretation results
# =========================================================
df_feature_blocks = pd.DataFrame(feature_block_mat, index=[s for s in subsID_1 if s in interp_feature_blocks], columns=FEATURE_BLOCK_NAMES)
df_feature_blocks.index.name = "Subject_ID"
df_feature_blocks.to_csv("interpret_feature_blocks_total.csv")

df_feature_blocks_norm = pd.DataFrame(feature_block_norm_mat, index=[s for s in subsID_1 if s in interp_feature_blocks], columns=FEATURE_BLOCK_NAMES)
df_feature_blocks_norm.index.name = "Subject_ID"
df_feature_blocks_norm.to_csv("interpret_feature_blocks_normalized.csv")

df_individual_features = pd.DataFrame(individual_feature_mat, index=[s for s in subsID_1 if s in interp_individual_features], columns=INDIV_FEATURE_NAMES)
df_individual_features.index.name = "Subject_ID"
df_individual_features.to_csv("interpret_individual_features.csv")

df_roi_nodes = pd.DataFrame(roi_node_mat, index=[s for s in subsID_1 if s in interp_roi_nodes], columns=ROI_NAMES)
df_roi_nodes.index.name = "Subject_ID"
df_roi_nodes.to_csv("interpret_roi_nodes_total.csv")

df_roi_nodes_norm = pd.DataFrame(roi_node_norm_mat, index=[s for s in subsID_1 if s in interp_roi_nodes_norm], columns=ROI_NAMES)
df_roi_nodes_norm.index.name = "Subject_ID"
df_roi_nodes_norm.to_csv("interpret_roi_nodes_normalized.csv")

df_agg_nodes = pd.DataFrame(agg_node_mat, index=[s for s in subsID_1 if s in interp_agg_nodes], columns=ROI_NAMES)
df_agg_nodes.index.name = "Subject_ID"
df_agg_nodes.to_csv("interpret_agg_nodes.csv")

df_within_roi_edges = pd.DataFrame(within_roi_edge_mat, index=[s for s in subsID_1 if s in interp_within_roi_edges], columns=ROI_NAMES)
df_within_roi_edges.index.name = "Subject_ID"
df_within_roi_edges.to_csv("interpret_within_roi_edges_total.csv")

df_within_roi_edges_norm = pd.DataFrame(within_roi_edge_norm_mat, index=[s for s in subsID_1 if s in interp_within_roi_edges_norm], columns=ROI_NAMES)
df_within_roi_edges_norm.index.name = "Subject_ID"
df_within_roi_edges_norm.to_csv("interpret_within_roi_edges_normalized.csv")

df_roi_to_agg_edges = pd.DataFrame(roi_to_agg_edge_mat, index=[s for s in subsID_1 if s in interp_roi_to_agg_edges], columns=ROI_NAMES)
df_roi_to_agg_edges.index.name = "Subject_ID"
df_roi_to_agg_edges.to_csv("interpret_roi_to_agg_edges_total.csv")

df_roi_to_agg_edges_norm = pd.DataFrame(roi_to_agg_edge_norm_mat, index=[s for s in subsID_1 if s in interp_roi_to_agg_edges_norm], columns=ROI_NAMES)
df_roi_to_agg_edges_norm.index.name = "Subject_ID"
df_roi_to_agg_edges_norm.to_csv("interpret_roi_to_agg_edges_normalized.csv")

for s in subsID_1:
    if s in interp_inter_agg_edges:
        df_inter = pd.DataFrame(interp_inter_agg_edges[s] * 100, index=ROI_NAMES, columns=ROI_NAMES)
        df_inter.to_csv(f"interpret_inter_agg_edges_subject_{s}.csv")

pd.DataFrame({"Importance": group_feature_blocks}, index=FEATURE_BLOCK_NAMES).to_csv("group_feature_blocks_total.csv")
pd.DataFrame({"Importance": group_feature_blocks_norm}, index=FEATURE_BLOCK_NAMES).to_csv("group_feature_blocks_normalized.csv")
pd.DataFrame({"Importance": group_individual_features, "SEM": group_individual_features_sem}, index=INDIV_FEATURE_NAMES).to_csv("group_individual_features.csv")

pd.DataFrame({"Importance": group_roi_nodes}, index=ROI_NAMES).to_csv("group_roi_nodes_total.csv")
pd.DataFrame({"Importance": group_roi_nodes_norm}, index=ROI_NAMES).to_csv("group_roi_nodes_normalized.csv")

pd.DataFrame({"Importance": group_agg_nodes}, index=ROI_NAMES).to_csv("group_agg_nodes.csv")

pd.DataFrame({"Importance": group_within_roi_edges}, index=ROI_NAMES).to_csv("group_within_roi_edges_total.csv")
pd.DataFrame({"Importance": group_within_roi_edges_norm}, index=ROI_NAMES).to_csv("group_within_roi_edges_normalized.csv")

pd.DataFrame({"Importance": group_roi_to_agg_edges}, index=ROI_NAMES).to_csv("group_roi_to_agg_edges_total.csv")
pd.DataFrame({"Importance": group_roi_to_agg_edges_norm}, index=ROI_NAMES).to_csv("group_roi_to_agg_edges_normalized.csv")

pd.DataFrame(group_inter_agg_edges, index=ROI_NAMES, columns=ROI_NAMES).to_csv("group_inter_agg_edges.csv")


# =========================================================
# Plot all interpretation outputs
# =========================================================
plot_boxplot_with_jitter(feature_block_mat, FEATURE_BLOCK_NAMES,
                         "Feature-block importance (total)")

plot_boxplot_with_jitter(feature_block_norm_mat, FEATURE_BLOCK_NAMES,
                         "Feature-block importance (normalized per feature)")

plot_individual_feature_importance_with_sem(group_individual_features,
                                            group_individual_features_sem,
                                            INDIV_FEATURE_NAMES,
                                            INDIV_FEATURE_GROUPS,
                                            "Individual-feature importance with SEM")

plot_boxplot_with_jitter(roi_node_mat, ROI_NAMES,
                         "ROI node importance (total)")

plot_boxplot_with_jitter(roi_node_norm_mat, ROI_NAMES,
                         "ROI node importance (normalized by number of ROI nodes)")

plot_boxplot_with_jitter(agg_node_mat, ROI_NAMES,
                         "Aggregation node importance")

plot_boxplot_with_jitter(within_roi_edge_mat, ROI_NAMES,
                         "Within-ROI edge importance (total)")

plot_boxplot_with_jitter(within_roi_edge_norm_mat, ROI_NAMES,
                         "Within-ROI edge importance (normalized by number of edges)")

plot_boxplot_with_jitter(roi_to_agg_edge_mat, ROI_NAMES,
                         "ROI-to-aggregation edge importance (total)")

plot_boxplot_with_jitter(roi_to_agg_edge_norm_mat, ROI_NAMES,
                         "ROI-to-aggregation edge importance (normalized by number of edges)")

plot_heatmap(group_inter_agg_edges, ROI_NAMES,
             "Inter-aggregation edge importance")
