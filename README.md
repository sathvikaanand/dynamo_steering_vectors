#dynamo_steering_vectors

based on https://arxiv.org/abs/2205.05124!


run: 

1. python process_csv.py: This reads from out_CID.csv, collects a sample of 20 SMILES strings with the highest binding affinities. Saved to sample.parquet

2. python steering.py OR python steeringv2.py: steering.py uses HuggingFace GPT2 models & tokenizer, steering.py uses Acegen GPT2. This reads sample.parquet and calculates 8 steering vectors for each SMILES string and saves it in plot_sample_vectors.parquet

3. python plot.py: plots all 8 steering vectors, color coded, on one TSNE plot. optionally can print the BLEU scores for each. 