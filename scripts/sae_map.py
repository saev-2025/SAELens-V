import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt
from sae_lens.activation_visualization import load_sae


sae_path = ""
sae_device = "cuda:0"
sae = load_sae(sae_path, sae_device)

features = sae.W_dec.cpu().detach().numpy()  # [65536, 4096]

act_data_path = ""
activation_counts = np.zeros(features.shape[0], dtype=int)

with open(act_data_path, "r") as f:
    for line in f:
        index, count = line.strip().split(":")
        activation_counts[int(index)] = int(count)

active_indices = np.where(activation_counts > 0)[0]
inactive_indices = np.where(activation_counts == 0)[0]

sample_size = 70000
if sample_size > len(inactive_indices):
    sample_size = len(inactive_indices) 
sampled_inactive_indices = np.random.choice(inactive_indices, size=sample_size, replace=False)

selected_indices = np.concatenate([active_indices, sampled_inactive_indices])
selected_features = features[selected_indices]
selected_activation_counts = activation_counts[selected_indices]

from sklearn.preprocessing import normalize
normalized_features = normalize(selected_features)

cos_sim_matrix = cosine_similarity(normalized_features)

distance_matrix = 1 - cos_sim_matrix 
distance_matrix = np.clip(distance_matrix, 0, None) 

reducer = umap.UMAP(
    n_components=2,  
    metric="precomputed",  
    n_neighbors=15, 
    min_dist=0.1,
    n_jobs=64,
    verbose=True,  
    low_memory=True, 
)
embedded_points = reducer.fit_transform(distance_matrix)

sorted_indices = np.argsort(selected_activation_counts) 
embedded_points = embedded_points[sorted_indices]
sorted_activation_counts = selected_activation_counts[sorted_indices]

csv_output_path = ""
np.savetxt(
    csv_output_path,
    np.column_stack((embedded_points, selected_activation_counts)),
    delimiter=",",
    header="Component 1,Component 2,Activation Counts",
    comments=""
)
print(f"UMAP result saves to {csv_output_path}")

plt.figure(figsize=(10, 8))

colors = sorted_activation_counts
sizes = np.where(sorted_activation_counts > 0, 20, 5) 

scatter = plt.scatter(
    embedded_points[:, 0], 
    embedded_points[:, 1], 
    c=colors, 
    cmap="viridis", 
    s=sizes, 
    alpha=0.7
)
plt.colorbar(scatter, label="Activation Counts")
plt.title("UMAP Visualization with Sampled Inactive Points")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")

image_output_path = ""
plt.savefig(image_output_path, format="png", dpi=300)

plt.show()
