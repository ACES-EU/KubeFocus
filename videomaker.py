import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import os
from datetime import datetime, timedelta
import time
import joblib
from tqdm import tqdm


#parameters
root_dir = '/home/sascha/kubernetes-intrusion-detection-main/KubeFocus'
output_dir=os.path.join(root_dir,'video')
os.makedirs(output_dir, exist_ok=True)
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('----------loading Metrics Data----------')
csv_file = f'{root_dir}/datasets/all_datasets_rf_ts.csv'

data = pd.read_csv(csv_file)
data = data[data['attack'] == 0]
if 'timestamp' in data.columns:
    data = data.drop(columns=['timestamp'])

data_metrics=data.iloc[:, np.r_[1:14, 19:384]]
data_metrics = data_metrics.select_dtypes(include=[np.number])



data_metrics = data_metrics.fillna(0)


metrics_scaler = MinMaxScaler()
scaled_data_metrics = metrics_scaler.fit_transform(data_metrics)

X_scaled_metric = torch.tensor(scaled_data_metrics , dtype=torch.float32)
X_tensor_metric=X_scaled_metric


class DualAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualAttentionAutoencoder, self).__init__()
        
        # Pairwise Attention mechanism: Learnable pairwise weights between features
        self.pairwise_attention_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [n_features, n_features]
        nn.init.xavier_uniform_(self.pairwise_attention_weights)  # Initialize
        
        # Per-Sample Attention mechanism: Attention per feature for each sample
        self.per_feature_attention_weights = nn.Parameter(torch.Tensor(input_dim))  # [n_features]
        nn.init.uniform_(self.per_feature_attention_weights, a=0.0, b=1.0)  # Initialize
        
        # Encoder with additional layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU()
        )
        
        # Decoder with additional layer
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pairwise attention mechanism
        pairwise_attention_scores = torch.matmul(x, self.pairwise_attention_weights) 
        pairwise_attention_coeffs = torch.softmax(pairwise_attention_scores, dim=-1) 
        pairwise_attention_matrix = torch.matmul(pairwise_attention_coeffs.T, pairwise_attention_coeffs)
        
        # Per-sample attention mechanism
        per_feature_attention_coeffs = torch.softmax(self.per_feature_attention_weights, dim=0) 
        attended_input = x * per_feature_attention_coeffs
        
        # Encoding and decoding with added layers
        encoded = self.encoder(attended_input)
        reconstructed = self.decoder(encoded)
        
        return reconstructed, pairwise_attention_matrix, per_feature_attention_coeffs

# Train the autoencoder with correct final attention matrices storage
epochs = 100 # Number of epochs

input_dim = X_tensor_metric.shape[1]
hidden_dim = 64
model = DualAttentionAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

metrics_pairwise_attention_matrix = None
metrices_per_feature_attention = None

print('----------training metrics camera----------')
for epoch in range(epochs):
    model.train()  
    
    optimizer.zero_grad() 
    
    reconstructed, pairwise_attention_matrix, per_feature_attention_coeffs = model(X_tensor_metric)
    
    loss = criterion(reconstructed, X_tensor_metric)
    
    loss.backward()
    optimizer.step()
    
    if epoch == epochs - 1:
        metrics_pairwise_attention_matrix = pairwise_attention_matrix.detach().cpu().numpy()
        metrices_per_feature_attention = per_feature_attention_coeffs.detach().cpu().numpy()
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


print("Final Pairwise Attention Matrix Shape:", metrics_pairwise_attention_matrix.shape)
print("Final Per-feature Attention Matrix Shape:", metrices_per_feature_attention.shape)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
feature_embeddings_2d = tsne.fit_transform(metrics_pairwise_attention_matrix)  # Shape: [n_features, 2]

grid_size = 32
tsne_min = np.min(feature_embeddings_2d, axis=0)
tsne_max = np.max(feature_embeddings_2d, axis=0)
file_prefix = 'metrics'
tsne_scaled = (feature_embeddings_2d - tsne_min) / (tsne_max - tsne_min)
tsne_grid_metrics = np.floor(tsne_scaled * (grid_size - 1)).astype(int)  # Shape: [n_features, 2]


print('----------loading Network Data----------')

data_network=data.iloc[:,15:19]

network_scaler = MinMaxScaler()
scaled_data_network = network_scaler.fit_transform(data_network)

X_scaled_traffic = torch.tensor(scaled_data_network, dtype=torch.float32)
X_tensor_traffic=X_scaled_traffic


input_dim = X_tensor_traffic.shape[1]
hidden_dim = 64  
model = DualAttentionAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

network_pairwise_attention_matrix = None
network_per_feature_attention = None

print('----------training Network camera----------')

for epoch in range(epochs):
    model.train() 
    
    optimizer.zero_grad()  
    
    reconstructed, pairwise_attention_matrix, per_feature_attention_coeffs = model(X_tensor_traffic)
    
 
    loss = criterion(reconstructed, X_tensor_traffic)
    
    loss.backward()
    optimizer.step()
    
    if epoch == epochs - 1:
        network_pairwise_attention_matrix = pairwise_attention_matrix.detach().cpu().numpy()
        network_per_feature_attention = per_feature_attention_coeffs.detach().cpu().numpy()
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


print("Final Pairwise Attention Matrix Shape:", network_pairwise_attention_matrix.shape)
print("Final Per-feature Attention Matrix Shape:", network_per_feature_attention.shape)

tsne = TSNE(n_components=2, perplexity=2, random_state=42)
feature_embeddings_2d = tsne.fit_transform(network_pairwise_attention_matrix)  # Shape: [n_features, 2]

grid_size = 32
tsne_min = np.min(feature_embeddings_2d, axis=0)
tsne_max = np.max(feature_embeddings_2d, axis=0)
tsne_scaled = (feature_embeddings_2d - tsne_min) / (tsne_max - tsne_min)
tsne_grid_network = np.floor(tsne_scaled * (grid_size - 1)).astype(int)  # Shape: [n_features, 2]


print('----------loading Log Data----------')

data['Pod_Logs'] = data['Pod_Logs'].fillna(0).astype('str')

pod_mapping = {'node_k8s-master-1': 1, 'node_k8s-worker-1': 2, 'node_k8s-worker-2': 3}
data['pod_label'] = data[['node_k8s-master-1', 'node_k8s-worker-1', 'node_k8s-worker-2']].idxmax(axis=1).map(pod_mapping)

tfidf = TfidfVectorizer(norm='l2')
tfidf_matrix = tfidf.fit_transform(data['Pod_Logs'])
tfidf_array = tfidf_matrix.toarray()  # Convert to array

tfidf_array = np.nan_to_num(tfidf_array, nan=0, posinf=1, neginf=0)

data_list = []

for i in range(len(tfidf_array)):
    pod_label = data.iloc[i]['pod_label']
    embedding = torch.tensor(tfidf_array[i], dtype=torch.float32)

    if torch.isnan(embedding).any():
        print(f"Warning: NaN found in embedding at index {i}, replacing with zero tensor")
        embedding = torch.zeros_like(embedding)

    data_list.append((pod_label, embedding))
    if torch.isnan(embedding).any():
        print(f"Warning: NaN found in embedding at index {i}, replacing with zero tensor")
        embedding = torch.zeros_like(embedding)
    
    data_list.append((pod_label, embedding))

logs_embeddings=data_list

print('log list created. length of log list is:',len(data_list))

metrics_attention = metrices_per_feature_attention / metrices_per_feature_attention.max()
traffic_attention = network_per_feature_attention/ network_per_feature_attention.max()

def scale_to_255(df, columns_to_exclude=None):
    df_scaled = df.copy()
    if not columns_to_exclude==None:
        for col in df.columns:
            if col not in columns_to_exclude:
                min_val = df[col].min()
                max_val = df[col].max()
                df_scaled[col] = 255 * (df[col] - min_val) / (max_val - min_val)
    else:
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df_scaled[col] = 255 * (df[col] - min_val) / (max_val - min_val)

    return df_scaled

data = pd.read_csv(csv_file)
data_metrics=data.iloc[:, np.r_[1:14, 19:384]]
data_metrics = data_metrics.select_dtypes(include=[np.number])

data_network = data.iloc[:, 15:19]
data_network['attack'] = data['attack']
data_network['attack'] = data['attack'].apply(lambda x: 0 if x == 0 else 1)
print(data_network['attack'].value_counts())


traffic_df_scaled = scale_to_255(data_network, columns_to_exclude=['attack'])
metrics_df_scaled = scale_to_255(data_metrics)
print(traffic_df_scaled['attack'].value_counts())

pod_labels = [entry[0] for entry in logs_embeddings]
min_pod_label, max_pod_label = min(pod_labels), max(pod_labels)


def scale_pod_label(value):
    """Scale the pod_label to the range 0-255."""
    return 255 * (value - min_pod_label) / (max_pod_label - min_pod_label)

logs_embeddings_scaled = []
for entry in logs_embeddings:
    pod_label,embedding = entry
    scaled_pod_label = scale_pod_label(pod_label)
    embedding = embedding.float()
    embedding_min, embedding_max = embedding.min(), embedding.max()
    scaled_embedding = 255 * (embedding - embedding_min) / (embedding_max - embedding_min)
    scaled_embedding = torch.nan_to_num(scaled_embedding, nan=0.0, posinf=255.0, neginf=0.0)
    logs_embeddings_scaled.append((scaled_pod_label, scaled_embedding))


print('----------Saving all images into one .pt file----------')

# Use the smallest common length
min_len = min(len(traffic_df_scaled), len(metrics_df_scaled), len(logs_embeddings_scaled))

all_data = []

for idx in tqdm(range(min_len), desc="Processing images"):
    traffic_sample = traffic_df_scaled.iloc[idx].drop(['attack']).values
    metrics_sample = metrics_df_scaled.iloc[idx].values
    logs_embedding = logs_embeddings_scaled[idx][1].numpy()
    scaled_pod_label = logs_embeddings_scaled[idx][0]
    label = int(traffic_df_scaled.iloc[idx]['attack'])  # get the label

    image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    # Red channel: traffic
    traffic_sample = np.nan_to_num(traffic_sample)
    for feature_idx, (x, y) in enumerate(tsne_grid_network):
        image[x, y, 0] = np.clip(traffic_sample[feature_idx] * traffic_attention[feature_idx], 0, 255)

    # Green channel: metrics
    metrics_sample = np.nan_to_num(metrics_sample)
    for feature_idx, (x, y) in enumerate(tsne_grid_metrics):
        image[x, y, 1] = np.clip(metrics_sample[feature_idx] * metrics_attention[feature_idx], 0, 255)

    # Blue channel: logs
    for x in range(4):
        for y in range(4):
            image[x, y, 2] = np.clip(scaled_pod_label, 0, 255)

    feature_count = 0
    for x in range(4, grid_size):
        for y in range(grid_size):
            if feature_count < len(logs_embedding):
                value = logs_embedding[feature_count]
                if not np.isfinite(value):
                    value = 0
                image[x, y, 2] = np.clip(value, 0, 255)
                feature_count += 1
            else:
                break
        if feature_count >= len(logs_embedding):
            break

    image_tensor = torch.tensor(image, dtype=torch.uint8)
    all_data.append({'image': image_tensor, 'label': label})

# Save everything into one .pt file
torch.save(all_data, os.path.join(output_dir, 'image_dataset.pt'))
artifacts_dir = os.path.join(root_dir, 'artifacts')
os.makedirs(artifacts_dir, exist_ok=True)

joblib.dump(metrics_scaler, os.path.join(artifacts_dir, 'metrics_scaler.pkl'))
joblib.dump(network_scaler, os.path.join(artifacts_dir, 'network_scaler.pkl'))
torch.save(torch.tensor(traffic_attention), os.path.join(artifacts_dir, 'traffic_attention.pt'))
torch.save(torch.tensor(metrics_attention), os.path.join(artifacts_dir, 'metrics_attention.pt'))

# Save t-SNE grids
np.save(os.path.join(artifacts_dir, 'tsne_grid_network.npy'), tsne_grid_network)
np.save(os.path.join(artifacts_dir, 'tsne_grid_metrics.npy'), tsne_grid_metrics)

# Save TF-IDF vectorizer for logs (if needed during live inference)
joblib.dump(tfidf, os.path.join(artifacts_dir, 'tfidf_vectorizer.pkl'))
print(f"Saved {len(all_data)} image-label pairs to {os.path.join(output_dir, 'image_dataset.pt')}")
