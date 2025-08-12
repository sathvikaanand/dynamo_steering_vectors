import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the CSV file containing steering vectors
input_file = "/Users/sathv/dynamo/plot_random_sample_vectors.parquet"
df = pd.read_parquet(input_file)
print(df.columns)

# Ensure the 'steering_vector' column exists
if 'steering_vector' not in df.columns:
    raise ValueError("The input file must contain a 'steering_vector' column.")
# Convert the steering vectors from string to numpy arrays
# df['steering_vector'] = df['steering_vector'].apply(lambda x: np.array(eval(x)))

# Stack the steering vectors into a 2D array
# vectors = np.stack(df['steering_vector'].values)

# # Apply t-SNE for dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(vectors)

# # Plot the t-SNE results
# plt.figure(figsize=(10, 8))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
# plt.title("t-SNE Plot of Steering Vectors")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.show()

df_filtered = df[df['bleu_score'] > 0]


data = np.vstack(df_filtered['steering_vector'].values)
all_labels = df_filtered['SMILES'].values
tsne = TSNE(n_components=2, random_state=0, perplexity=5, verbose=1)
proj = tsne.fit_transform(data)

# Plot, color by SMILES index
plt.figure(figsize=(8,6))
unique = list(dict.fromkeys(all_labels))
colors = plt.cm.tab20(np.linspace(0,1,len(unique)))
for i, smi in enumerate(unique):
    idxs = [j for j,label in enumerate(all_labels) if label==smi]
    plt.scatter(proj[idxs,0], proj[idxs,1], c=[colors[i]], label=f"SMILES {i}")

# for i, bleu in enumerate(df_filtered['bleu_score']):
#     plt.text(proj[i, 0], proj[i, 1], f"{bleu:.2f}", fontsize=8, color='red')

plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
plt.title('t-SNE of 8 seeds for each SMILES')
plt.tight_layout()
plt.show()