# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:36:51 2026

@author: JEsmaelpoor
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr

# Subject IDs
subsID_1 = [1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 41, 43, 44]

# Folder containing graphs
data_folder = "C:/Melbourne University/PhD studies_works/Study Five-GNN/NHMRC Project/final codes/sess1_graphs/"


# Define GAT Model
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


# Function to Load Graphs
def load_all_graphs(folder, subject_list):
    all_graphs = []
    for subject in subject_list:
        for file in os.listdir(folder):
            if file.startswith(f"sub_{subject}_") and file.endswith(".pt"):
                graph = torch.load(os.path.join(folder, file), weights_only=False)
                all_graphs.append(graph)
    return all_graphs


# Training Function
def train_gat(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

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


# Testing Function
@torch.no_grad()
def test_gat(model, loader, criterion, device):
    model.eval()
    total_loss = 0
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


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Leave-One-Subject-Out Validation
results = {}

for test_subject in subsID_1:
    print(f"\nTesting on subject {test_subject}")

    # Load test graphs
    test_graphs = load_all_graphs(data_folder, [test_subject])

    # Load training graphs
    train_subjects = [s for s in subsID_1 if s != test_subject]
    train_graphs = load_all_graphs(data_folder, train_subjects)

    # Create DataLoaders with real minibatches
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Initialize model
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

    # Optimizer, scheduler, loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = torch.nn.HuberLoss(delta=1.0)

    # Train
    epochs = 101
    train_losses = []
    
    best_train_loss = float("inf")
    patience = 15
    min_improvement = 8e-5
    # min_improvement = 7.5e-5
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_gat(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        scheduler.step()
    
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
    
        # Early stopping based on training loss
        if best_train_loss - train_loss > min_improvement:
            best_train_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
    
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    # # Plot training loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(epochs), train_losses, label="Train Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title(f"Training Loss - Subject {test_subject}")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # Test
    test_loss, avg_predicted_score, avg_true_score = test_gat(model, test_loader, criterion, device)

    # Store results
    results[test_subject] = {
        "MSE": test_loss,
        "Predicted_BKB": avg_predicted_score,
        "True_BKB": avg_true_score
    }

    print(f"Subject {test_subject}: Test Loss = {test_loss:.4f}")
    print(f"Predicted BKB = {avg_predicted_score:.2f}, True BKB = {avg_true_score:.2f}")


# Save results to CSV
df = pd.DataFrame.from_dict(results, orient="index")
df.to_csv("graph_level_results_bkb_sideFedToFC.csv", index_label="Subject_ID")


# Correlation plot
true_scores = df["True_BKB"].values
pred_scores = df["Predicted_BKB"].values

r, p = pearsonr(true_scores, pred_scores)

plt.figure(figsize=(7, 7))
plt.scatter(true_scores, pred_scores)

# Regression line
m, b = np.polyfit(true_scores, pred_scores, 1)
x_line = np.linspace(true_scores.min(), true_scores.max(), 100)
plt.plot(x_line, m * x_line + b)

plt.xlabel("True BKB")
plt.ylabel("Predicted BKB")
plt.title(f"Predicted vs True BKB\nR = {r:.3f}, p = {p:.4g}")
plt.grid(True)
plt.tight_layout()
plt.show()