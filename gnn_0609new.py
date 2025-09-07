import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, roc_auc_score
import warnings

# 忽略ROC AUC计算时的特定警告
warnings.filterwarnings("ignore", message=".*only one class.*", category=UserWarning)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load and Process Data
# Load protein labels
df_labels = pd.read_csv('output/pfam_SSCI/prot_id_label.tsv', sep='\t')
df_network = pd.read_csv('output/pfam_SSCI/network.tsv', sep='\t')

# Get all protein IDs and create node index mapping
prot_ids = df_labels['prot_id'].tolist()
prot_id_to_idx = {pid: idx for idx, pid in enumerate(prot_ids)}
num_nodes = len(prot_ids)

# Process labels: list of labels per node
labels_per_node = []
for _, row in df_labels.iterrows():
    if pd.isna(row['uniprotKB_pfam']) or row['uniprotKB_pfam'].strip() == '':
        labels_per_node.append([])
    else:
        labels = row['uniprotKB_pfam'].split(';')
        labels = [label.strip() for label in labels if label.strip()]
        labels_per_node.append(labels)

# Collect unique labels and map to indices
all_labels = set()
for labels in labels_per_node:
    all_labels.update(labels)
all_labels = sorted(list(all_labels))
num_labels = len(all_labels)
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Create target tensor (num_nodes x num_labels)
target = torch.zeros(num_nodes, num_labels, dtype=torch.float)
labeled_nodes = []
for i, labels in enumerate(labels_per_node):
    if labels:
        labeled_nodes.append(i)
        for label in labels:
            target[i, label_to_idx[label]] = 1
unlabeled_nodes = [i for i in range(num_nodes) if i not in labeled_nodes]

# 2. Construct Graph
edge_index = []
edge_weight = []
for _, row in df_network.iterrows():
    src = prot_id_to_idx[row['Source_prot_id']]
    tgt = prot_id_to_idx[row['Target_prot_id']]
    weight = row['alntmscore']
    edge_index.append([src, tgt])
    edge_index.append([tgt, src])  # Undirected graph
    edge_weight.extend([weight, weight])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)
data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes).to(device)

# 3. Split Labeled Nodes (8:1:1)
train_val_nodes, test_nodes = train_test_split(labeled_nodes, test_size=0.1, random_state=42)
train_nodes, val_nodes = train_test_split(train_val_nodes, test_size=1 / 9,
                                          random_state=42)  # 8:1 ratio after test split

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_nodes] = True
val_mask[val_nodes] = True
test_mask[test_nodes] = True

target = target.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)


# 4. Define GNN Model
class GNN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, num_labels):
        super(GNN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.gcn1 = GCNConv(embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, node_indices, edge_index, edge_weight):
        x = self.embedding(node_indices)
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.gcn2(x, edge_index, edge_weight))
        x = self.out(x)
        return x


# Initialize model
model = GNN(num_nodes=num_nodes, embedding_dim=128, hidden_dim=256, num_labels=num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Compute pos_weight for label imbalance
num_pos = target[train_mask].sum(dim=0)
pos_weight = (len(train_nodes) - num_pos) / (num_pos + 1e-6)  # Avoid division by zero
pos_weight = torch.clamp(pos_weight, max=100).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# 定义评估函数
def evaluate(y_true, y_pred_probs, threshold=0.5):
    """
    评估多标签分类性能
    返回包含多个指标的字典
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_binary = (y_pred_probs > threshold).float().cpu().numpy()

    metrics = {}

    # 精确率、召回率、F1分数 (宏平均和微平均)
    metrics['precision_macro'] = precision_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true_np, y_pred_binary, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true_np, y_pred_binary, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true_np, y_pred_binary, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true_np, y_pred_binary, average='micro', zero_division=0)

    # 汉明损失
    metrics['hamming_loss'] = hamming_loss(y_true_np, y_pred_binary)

    # 子集准确率 (完全匹配)
    metrics['subset_accuracy'] = (y_pred_binary == y_true_np).all(axis=1).mean()

    # 尝试计算ROC AUC (可能失败)
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true_np, y_pred_probs.cpu().numpy(), average='macro')
    except:
        metrics['roc_auc_macro'] = float('nan')

    try:
        metrics['roc_auc_micro'] = roc_auc_score(y_true_np, y_pred_probs.cpu().numpy(), average='micro')
    except:
        metrics['roc_auc_micro'] = float('nan')

    return metrics


# 5. Training Loop
node_indices = torch.arange(num_nodes, device=device)
best_val_f1 = 0.0

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(node_indices, data.edge_index, data.edge_attr)
    loss = criterion(out[train_mask], target[train_mask])
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = out[val_mask]
        val_pred_probs = torch.sigmoid(val_out)
        val_metrics = evaluate(target[val_mask], val_pred_probs)

        # 使用宏平均F1作为主要指标
        current_val_f1 = val_metrics['f1_macro']

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            # 保存最佳模型
            best_model_state = model.state_dict().copy()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(f"Validation Metrics:")
            for k, v in val_metrics.items():
                print(f"  {k:20}: {v:.4f}")

# 加载最佳模型
model.load_state_dict(best_model_state)

# 6. Evaluate on Test Set
model.eval()
with torch.no_grad():
    test_out = model(node_indices, data.edge_index, data.edge_attr)[test_mask]
    test_pred_probs = torch.sigmoid(test_out)
    test_metrics = evaluate(target[test_mask], test_pred_probs)

    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:20}: {v:.4f}")

# 7. Predict for Unlabeled Nodes
unlabeled_predictions = []
model.eval()
with torch.no_grad():
    out = model(node_indices, data.edge_index, data.edge_attr)
    unlabeled_out = out[unlabeled_nodes]
    unlabeled_pred = (torch.sigmoid(unlabeled_out) > 0.5).float()
    for i, pred_row in enumerate(unlabeled_pred):
        pred_indices = torch.where(pred_row == 1)[0].tolist()
        pred_labels = [idx_to_label[idx] for idx in pred_indices]
        prot_id = prot_ids[unlabeled_nodes[i]]
        unlabeled_predictions.append((prot_id, pred_labels))

# Output predictions
with open('unlabeled_predictions.txt', 'w') as f:
    for prot_id, labels in unlabeled_predictions:
        labels_str = ';'.join(labels) if labels else 'None'
        f.write(f"{prot_id}\t{labels_str}\n")
print("Predictions for unlabeled proteins saved to 'unlabeled_predictions.txt'")