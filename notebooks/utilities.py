import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
import string

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from collections import defaultdict, Counter

from matplotlib.colors import to_hex
import matplotlib.cm as cm

from plotly import graph_objects as go

import umap
from umap import UMAP
import hdbscan
from hdbscan import HDBSCAN
import community as community_louvain
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def get_data(input_path, output_path, n_component, min_cluster_size):
    embeddings_path = input_path
    embeddings = np.load(embeddings_path)
    umap_reducer = umap.UMAP(n_components=n_component, random_state=2023)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    np.save(output_path, cluster_labels)
    return reduced_embeddings, cluster_labels

def num_of_cluster(cluster_labels):
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters: {num_clusters}")
    
    
############################# preprocess function #########################################    
def preprocess(text):
   
    if not isinstance(text, str):
        return ""
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                               u"\U0001F700-\U0001F77F"  # Alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

####################################### get top k words using tf-idf ######################
def get_top_tfidf_words_per_cluster(df, cluster_labels, text_column, num_top_words=3):
    if len(df) != len(cluster_labels):
        raise ValueError("Length of dataframe and cluster labels must match.")

    df[text_column] = df[text_column].fillna('').replace([None], '')

    cluster_labels = np.array(cluster_labels)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    terms = tfidf_vectorizer.get_feature_names_out()
    
    top_words = defaultdict(list)
    for cluster in set(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        # Check if cluster_indices are within the valid range
        if len(cluster_indices) == 0:
            continue

        cluster_tfidf_matrix = tfidf_matrix[cluster_indices]
        mean_tfidf_scores = cluster_tfidf_matrix.mean(axis=0).A1
        top_indices = mean_tfidf_scores.argsort()[-num_top_words:][::-1]
        top_terms = [terms[i] for i in top_indices]
        top_words[cluster] = top_terms
    
    return top_words


#######################for plotting clusters using 2d map #################################
def plot_clusters(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=10000, metric='cosine', random_state=2023)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='Spectral', s=10)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title('TSNE visualization of clusters')
    plt.show()


###################for community detection#############################################
def build_cluster_graph(embeddings, labels, similarity_threshold):
    unique_labels = set(labels)
    unique_labels.discard(-1)  

    centroids = []
    for label in unique_labels:
        centroids.append(np.mean(embeddings[labels == label], axis=0))

    centroids = np.array(centroids)
    centroids = normalize(centroids, axis=1)
    cosine_matrix = np.dot(centroids, centroids.T)
    G = nx.Graph()
    for idx, label in enumerate(unique_labels):
        G.add_node(label, label=f"Cluster {label}")

    threshold = similarity_threshold
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i != j and cosine_matrix[i, j] > threshold:
                G.add_edge(label1, label2, weight=cosine_matrix[i, j])

    return G, cosine_matrix

def get_color_map(cosine_matrix):
    similarity_scores = np.mean(cosine_matrix, axis=1)
    norm_scores = (similarity_scores - np.min(similarity_scores)) / (np.max(similarity_scores) - np.min(similarity_scores))
    colors = [to_hex(cm.viridis(score)) for score in norm_scores]
    return colors

def annotate_graph_plotly(G, top_words, colors, communities):
    labels = {node: f"{G.nodes[node]['label']}:\n" + ", ".join(top_words[node]) for node in G.nodes}
    pos = {}
    community_dict = {}
    for node, community in communities.items():
        if community not in community_dict:
            community_dict[community] = []
        community_dict[community].append(node)

    angle_step = 2 * np.pi / len(community_dict)
    radius = 10  
    for idx, (community, nodes) in enumerate(community_dict.items()):
        angle = idx * angle_step
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        sub_pos = nx.spring_layout(G.subgraph(nodes), center=(center_x, center_y))
        pos.update(sub_pos)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(136, 136, 136, 0.7)'),  
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(labels[node])
        node_color.append(colors[communities[node]])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Cluster Similarity Graph with Annotations',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    fig.update_layout(height=800)
    fig.show()

def visualize_clusters_with_communities(reduced_embeddings, cluster_labels, top_words, similarity_threshold):
    G, cosine_matrix = build_cluster_graph(reduced_embeddings, cluster_labels, similarity_threshold)

    # Run Louvain community detection algorithm
    partition = community_louvain.best_partition(G)
    communities = {node: partition[node] for node in G.nodes}
    print(f"unique communities are: {communities}")
    value_counts = Counter(communities.values())
    print(value_counts)

    unique_communities = list(set(communities.values()))
    color_palette = cm.get_cmap('tab20', len(unique_communities))
    community_colors = [to_hex(color_palette(i)) for i in range(len(unique_communities))]
    community_colors_map = {unique_communities[i]: community_colors[i] for i in range(len(unique_communities))}
    node_colors = [community_colors_map[communities[node]] for node in G.nodes]

    annotate_graph_plotly(G, top_words, node_colors, communities)
    
    
########### Topic modelling ################################################
def bertopicModelling(n_components, min_cluster_size, top_n_words, data):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(n_components=n_components)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True, prediction_data=True)
    stopwords = list(stopwords.words('english'))
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)
    
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        language='english',
        calculate_probabilities=True,
        verbose=True
    )
    topics, probs = model.fit_transform(data)
    return model, topics, probs