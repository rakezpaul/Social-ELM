import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pycountry
from pytz import country_names
import re
import plotly.graph_objects as go
from sklearn.cluster import SpectralCoclustering
import numpy as np
import seaborn as sns

scopus = pd.read_csv("scopusELM.csv")
# print(scopus.head())

# #no. of publications per year
#
df = pd.DataFrame(scopus)
# print(df.head())
# year_counts = df['Year'].value_counts().sort_index()
# print(year_counts)
#
# # Plotting the data as a line chart
# plt.figure(figsize=(10, 6))
# plt.plot(year_counts.index, year_counts.values, marker='o', color='skyblue', linestyle='-')
# plt.title('Number of Occurrences Per Year')
# plt.xlabel('Year')
# plt.ylabel('Number of Occurrences')
# plt.xticks(year_counts.index)  # Set x-ticks to be the years
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xticks(year_counts.index)  # Ensure x-ticks are shown for the years
# plt.show()

#Funding & Affiliations
# funding = df[["FundingDetails", "Affiliations"]]
# # print(funding.head())

#directed graph

# G = nx.Graph()
# for index, row in df.iterrows(): G.add_edge(row['FundingDetails'], row['Affiliations'])
#
# plt.figure(figsize=(50, 50))  # Set the figure size
# pos = nx.spring_layout(G)       # Layout for a nice arrangement
# nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', alpha=0.7)
# plt.title("Network Diagram of Funding Agencies and Affiliations")
# plt.show()



#abstracts, title, authorkeywords
# speech_tag = df[["Title","Abstract", "AuthorKeywords"]]
# print(speech_tag.head())

#Countries
countries_set = {country.name: country.alpha_3 for country in pycountry.countries}

def extract_country(affiliations):
    if pd.isna(affiliations):
        return None

    for country in countries_set.keys():
        if country in affiliations:
            return country
    return None



df1 = df['country'] = df['affiliations'].apply(extract_country)
# print(df1)

#fundingagency
funding_agencies = df['fundingdetails']

#dataframe
df = pd.DataFrame(funding_agencies)

#cleaning function
def clean_agency(agency):
    if isinstance(agency, str):
        agency_cleaned = re.sub(r'\s*\(\d+\)|\d+', '', agency)
        return ' '.join(agency_cleaned.split()).strip()
    return ''

df2 = df['cleaned_fundingdetails'] = df['fundingdetails'].apply(clean_agency)
# print(df2)

#combine country and funding
combined_df = pd.concat([df1, df2], axis=1)
combined_df.columns = ['country', 'funder']

# print(combined_df)
df = combined_df[['country', 'funder']].dropna()

# Clean the data
# combined_df = combined_df.dropna(subset=['country', 'funder'])
# combined_df['country'] = combined_df['country'].astype(str)
# combined_df['funder'] = combined_df['funder'].astype(str)

# countries = combined_df['country'].unique().tolist()
# funders = combined_df['funder'].unique().tolist()

# #1 Create an empty graph
# G = nx.Graph()
#
# G.add_nodes_from([(c, {"bipartite": "country"}) for c in countries])
# G.add_nodes_from([(f, {"bipartite": "funder"}) for f in funders])
#
# # Add edges between country and funder
# for _, row in combined_df.iterrows():
#     G.add_edge(row['country'], row['funder'])
#
# # Plot
# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(G, k=0.5)
# nx.draw(G, pos, with_labels=False, node_color='skyblue', edge_color='gray', node_size=1000, font_size=8)
# plt.title("Country-Funder Network")
# plt.show()

#2 Cluster analysis

top_countries = (df['country'].value_counts().head(30).index)

top_funders = (df['funder'].value_counts().head(30).index)
print(top_funders)

df = df[df['country'].isin(top_countries) & df['funder'].isin(top_funders)]

#build a contingency matrix
matrix = (
    df
      .groupby(['country', 'funder'])
      .size()
      .unstack(fill_value=0)
      .astype(int)
)

#bicluster the matrix
n_row_clusters = 4          # try a few values
n_col_clusters = 4

model = SpectralCoclustering(
    n_clusters=min(n_row_clusters, n_col_clusters),
    random_state=42,
    svd_method="randomized"
)
model.fit(matrix)

# Re-order rows/columns by the cluster labels so blocks fall together
row_order = np.argsort(model.row_labels_)
col_order = np.argsort(model.column_labels_)
clustered = matrix.iloc[row_order, col_order]

# # visualization
# plt.figure(figsize=(12, 10))
# sns.heatmap(
#     clustered,
#     cmap="Blues",
#     cbar_kws={'label': 'Frequency'},
#     linewidths=0.05,
#     linecolor='lightgrey'
# )
#
# plt.title('Country-Funder biclusters')
# plt.ylabel('Countries (re-ordered by cluster)')
# plt.xlabel('Funders (re-ordered by cluster)')
# plt.tight_layout()
# plt.show()


# Create a bipartite graph
B = nx.Graph()

# Add nodes with bipartite attribute
countries = matrix.index.tolist()
funders = matrix.columns.tolist()

B.add_nodes_from(countries, bipartite='country')
B.add_nodes_from(funders, bipartite='funder')

# Add edges based on co-occurrence (only if frequency > 0)
for country in countries:
    for funder in funders:
        weight = matrix.loc[country, funder]
        if weight > 0:
            B.add_edge(country, funder, weight=weight)

# Frequency of each node (degree weighted by edge weights)
node_size = {
    node: sum(d['weight'] for _, _, d in B.edges(node, data=True))
    for node in B.nodes()
}

# Normalize for better sizing in plot
max_size = max(node_size.values())
scaled_node_size = {node: 300 + (3000 * size / max_size) for node, size in node_size.items()}

plt.figure(figsize=(14, 10))

# Use bipartite layout
from networkx.algorithms import bipartite

pos = nx.spring_layout(B, seed=42, k=0.3)  # Better separation than bipartite_layout

# Draw edges
nx.draw_networkx_edges(B, pos, alpha=0.3)

# Draw country nodes
country_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 'country']
nx.draw_networkx_nodes(
    B, pos,
    nodelist=country_nodes,
    node_color='skyblue',
    node_size=[scaled_node_size[n] for n in country_nodes],
    label='Countries'
)

# Draw funder nodes
funder_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 'funder']
nx.draw_networkx_nodes(
    B, pos,
    nodelist=funder_nodes,
    node_color='salmon',
    node_size=[scaled_node_size[n] for n in funder_nodes],
    label='Funders'
)

# Step 1: Get top 10 funders by total weight (degree)
top_10_funders = sorted(
    funder_nodes,
    key=lambda n: node_size[n],
    reverse=True
)[:10]

# Step 2: Build label dict
labels = {node: node for node in top_10_funders}


# Sort nodes by total weight
sorted_nodes = sorted(node_size.items(), key=lambda x: x[1], reverse=True)

# Get top 10 countries and funders separately
top_countries = [n for n, _ in sorted_nodes if n in country_nodes][:14]
top_funders = [n for n, _ in sorted_nodes if n in funder_nodes][:10]

# Combine and annotate
# top_labels = {n: n for n in top_countries + top_funders}
top_labels = {n: n for n in top_countries}

nx.draw_networkx_labels(B, pos, labels=top_labels, font_size=9, font_color='black')



plt.title("Country–Funder Network", fontsize=16)
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.show()

# Remove weak connections
B.remove_edges_from([(u, v) for u, v, d in B.edges(data=True) if d['weight'] < 2])
