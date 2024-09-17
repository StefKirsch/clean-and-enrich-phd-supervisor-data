# %% [markdown]
# ### Try modeling a network of authors - contributors

# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# %%
# !pip install networkx

# %%
df = pd.read_csv("data/cleaned/pubs.csv")

# %%
df = df.iloc[:100]

# %%
G = nx.Graph()

# %%
df.head()

# %%
for author in df['author_name'].unique():
    G.add_node(author, node_type='author')

# %%
for index, row in df.iterrows():
    for col in range(1, 11):  # Assuming there are 10 contributor columns
        contributor = row[f'contributor_{col}']
        if not pd.isnull(contributor):
            G.add_node(contributor, node_type='contributor')

# %%
for index, row in df.iterrows():
    author = row['author_name']
    for col in range(1, 11):  # Assuming there are 10 contributor columns
        contributor = row[f'contributor_{col}']
        if not pd.isnull(contributor):
            G.add_edge(author, contributor)

# %%
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


# %%
print("Nodes:", G.nodes())
print("Edges:", G.edges())


# %%
node = 'Azaripour, Adriano'
print("Neighbors of node", node, ":", list(G.neighbors(node)))


# %%
# Node attributes
print("Node attributes:", G.nodes(data=True))

# Edge attributes
print("Edge attributes:", G.edges(data=True))


# %%
columns_to_iterate = df.columns[7:].tolist()
val=[]
for column_name in columns_to_iterate:
    for value in df[column_name]:
        if pd.notna(value):
            if value not in val:
                val.append(value)
len(val)

# %%
vals=[]
for auth in df["author_name"]:
    if auth not in vals:
        vals.append(auth)
len(vals)

# %%
pos = nx.spring_layout(G, seed=42)
node_colors = {'author': 'blue', 'contributor': 'green'}
node_shapes = {'author': 'o', 'contributor': '^'}

# %%
# node_type = nx.get_node_attributes(G, 'node_type')
# node_color = [node_colors[node_type[node]] for node in G.nodes()]
# node_shape = [node_shapes[node_type[node]] for node in G.nodes()]

# %%
for node, data in G.nodes(data=True):
    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_colors[data['node_type']],
                           node_shape='o' if data['node_type'] == 'author' else '^', node_size=200)

# %%
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)

plt.show()

# %%
print(G)


