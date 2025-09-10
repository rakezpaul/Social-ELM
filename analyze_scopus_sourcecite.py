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


df = pd.DataFrame(scopus)
#source
source = df['source']

#cited
cited = df['cited_by']
# print(cited.head())

data = pd.DataFrame((source, cited))


# Define bins and labels
bins = [0, 50, 100, 200, float('inf')]
labels = ['0-50', '50-100', '100-200', '<200']

# Create a new column 'citation_group' to categorize the citations
df['citation_group'] = pd.cut(df['cited_by'], bins=bins, labels=labels, right=False)

# Group sources by citation categories and count them
grouped_sources = df.groupby('citation_group')['source'].agg(['count', list]).reset_index()
grouped_sources.columns = ['citation_group', 'source_count', 'sources']

# Prepare data for plotting
plot_data = df.copy()
# Add a column to define size for highlighting
plot_data['size'] = plot_data['cited_by'] * 0.1  # Scale size for visibility (you can adjust this factor)

# Create a scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(plot_data['cited_by'], plot_data['citation_group'].cat.codes,
                      s=plot_data['size'], alpha=0.7, c='blue')

# Highlight more cited sources with larger markers
plt.scatter(plot_data.loc[plot_data['cited_by'] > 100, 'cited_by'],
            plot_data.loc[plot_data['cited_by'] > 100, 'citation_group'].cat.codes,
            s=plot_data.loc[plot_data['cited_by'] > 100, 'size'],
            color='red', label='High citations (>100)', alpha=0.7)

# Customize the plot
plt.title('Scatter Plot of Sources by Citation Count')
plt.xlabel('Number of Citations')
plt.ylabel('Citation Group')
plt.yticks(range(len(labels)), labels)  # Set the y-ticks to the labels
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Define bins and labels
# bins = [0, 50, 100, 200, np.inf]
# labels = ['0-50', '50-100', '100-200', '200+']
#
# # Cut the citations into bins
# df['citation_group'] = pd.cut(df['cited_by'], bins=bins, labels=labels, right=False)

# # Generate coordinates for scatter plot (for visualization purposes)
# # You might want to customize this to reflect meaningful data points
# df['x'] = df['cited_by']  # Use citations as x-coordinate
# df['y'] = np.random.rand(len(df)) * 100  # Random y value for scatter plot
#
# # Create scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df['x'], df['y'], c=df['citation_group'].cat.codes, cmap='viridis', alpha=0.7)
#
# # Add colorbar
# plt.colorbar(ticks=range(len(labels)), label='Citation Groups',
#              format=plt.FuncFormatter(lambda x, _: labels[int(x)]))
#
# # Title and labels
# plt.title('Scatter Plot of Sources Based on Citation Groups')
# plt.xlabel('Citations')
# plt.ylabel('Random Y Value')
#
# # Remove x and y ticks (to not show labels)
# plt.xticks([])
# plt.yticks([])
#
# # Show the plot
# plt.show()

# Count occurrences and sum citations
