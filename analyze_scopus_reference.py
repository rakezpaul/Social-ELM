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

#title
# title = df['title']
# # print(title.head())

#references
references = df['references']


# data = pd.DataFrame((title, references))
# data_transpose = data.T
# print(data_transpose)


