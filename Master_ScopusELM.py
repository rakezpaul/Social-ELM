import pandas as pd
import spacy
import re
import networkx as nx
import matplotlib.pyplot as plt

scopus = pd.read_csv('scopusELM.csv')
df = pd.DataFrame(scopus)

df ['authorkeywords'] = df['authorkeywords'].fillna('')
df ['abstracts'] = df['abstract']

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    return text

df['processed_abstracts'] = df['abstract'].apply(preprocess_text)

from collections import defaultdict
from itertools import combinations

co_occurrence = defaultdict(int)

for _, row in df.iterrows():
    keywords = row['authorkeywords'].split(';')
    for keyword_pair in combinations(keywords, 2):
        co_occurrence[keyword_pair] += 1

co_occurrence_df = pd.DataFrame(co_occurrence.items(), columns=['Keyword Pair', 'Count'])

#NLP
nlp = spacy.load('en_core_web_sm')

def get_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df['entities'] = df['processed_abstracts'].apply(get_entities)

def get_pos_tags(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

df['pos_tags'] = df['processed_abstracts'].apply(get_pos_tags)

# Plot
import networkx as nx
import matplotlib.pyplot as plt

#Create a graph from the co-occurrence dictionary
G = nx.Graph()

for (k1, k2), count in co_occurrence.items():
    G.add_edge(k1, k2, weight=count)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False)
edges = nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] for u,v in G.edges()])
plt.show()

# Create co-occurrence matrix
# Calculate Centralities
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Print Centralities
print("Degree Centrality:")
print(degree_centrality)

print("Closeness Centrality:")
print(closeness_centrality)

print("Betweenness Centrality:")
print(betweenness_centrality)

# Calculate and Print Clustering Coefficient
global_clustering = nx.transitivity(G)
local_clustering = nx.clustering(G)

print("Global Clustering Coefficient:")
print(global_clustering)

print("Local Clustering Coefficients:")
print(local_clustering)

import community as community_louvain

# Detect communities
partition = community_louvain.best_partition(G)

# Print out the clusters
clusters = defaultdict(list)
for node, cluster_id in partition.items():
    clusters[cluster_id].append(node)

# Display clusters
for cluster_id, keywords in clusters.items():
    print(f"Cluster {cluster_id}: {keywords}")

# Calculate Degree Centrality for node sizes
degree_centrality = nx.degree_centrality(G)

# Extract co-occurrence counts for edges
edge_weights = nx.get_edge_attributes(G, 'weight')

# Define node sizes based on degree centrality
node_sizes = [v * 100 for v in degree_centrality.values()]  # Scale size to make it more visible

# Define edge widths based on co-occurrence counts
edge_weights_values = [edge_weights[edge] for edge in G.edges()]
edge_widths = [weight * 0.1 for weight in edge_weights_values]  # Scale edge thickness

# Colors for visibility
colors = ['skyblue' if centrality > 0.1 else 'orange' for centrality in degree_centrality.values()]

# Create layout for nodes
pos = nx.spring_layout(G)  # Fruchterman-Reingold force-directed algorithm

# Create the plot
plt.figure(figsize=(96, 80), dpi=150)  # Set figure size and dpi

# Draw nodes with custom sizes and colors
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.7)

# Draw edges with custom widths
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=0.1, )

# Title and display
plt.title("Keyword Co-Occurrence Network", size=5)
plt.axis('off')  # Hide the axes
plt.show()
