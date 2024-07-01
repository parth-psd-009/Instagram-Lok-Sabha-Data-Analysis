import numpy as np
import umap
import hdbscan
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
from matplotlib.colors import to_hex
from matplotlib import cm
from plotly import graph_objects as go
import community as community_louvain

class ClusterVisualizer:
    def __init__(self, embeddings_path, df_path):
        self.embeddings_path = embeddings_path
        self.df_path = df_path
        self.embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None
        self.df = None
        self.top_words = None
        self.G = None
        self.cosine_matrix = None
        self.communities = None

    def load_embeddings(self):
        self.embeddings = np.load(self.embeddings_path)

    def save_cluster_labels(self, output_path):
        np.save(output_path, self.cluster_labels)

    def cluster_embeddings(self, n_dimensions=128, min_cluster_size=34):
        umap_reducer = umap.UMAP(n_components=n_dimensions, random_state=2023)
        self.reduced_embeddings = umap_reducer.fit_transform(self.embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        self.cluster_labels = clusterer.fit_predict(self.reduced_embeddings)

    def load_dataframe(self, col_name):
        self.df = pd.read_csv(self.df_path)
        self.df[col_name] = self.df[col_name].fillna('')

    def get_top_tfidf_words_per_cluster(self, col_name, num_top_words=3, stop_words='english'):
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df[col_name])
        terms = tfidf_vectorizer.get_feature_names_out()
        top_words = defaultdict(list)

        for cluster in set(self.cluster_labels):
            cluster_indices = self.df[self.cluster_labels == cluster].index
            cluster_tfidf_matrix = tfidf_matrix[cluster_indices]
            mean_tfidf_scores = cluster_tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf_scores.argsort()[-num_top_words:][::-1]
            top_terms = [terms[i] for i in top_indices]
            top_words[cluster] = top_terms

        self.top_words = top_words

    def plot_tsne_clusters(self):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=10000, metric='cosine', random_state=2023)
        tsne_results = tsne.fit_transform(self.reduced_embeddings)
        
        plt.figure(figsize=(16, 10))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=self.cluster_labels, cmap='Spectral', s=10)
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.title('TSNE visualization of clusters')
        plt.show()

    def build_cluster_graph(self, similarity_threshold):
        unique_labels = set(self.cluster_labels)
        unique_labels.discard(-1)

        centroids = []
        for label in unique_labels:
            centroids.append(np.mean(self.reduced_embeddings[self.cluster_labels == label], axis=0))

        centroids = np.array(centroids)
        centroids = normalize(centroids, axis=1)
        self.cosine_matrix = np.dot(centroids, centroids.T)
        self.G = nx.Graph()
        for idx, label in enumerate(unique_labels):
            self.G.add_node(label, label=f"Cluster {label}")

        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i != j and self.cosine_matrix[i, j] > similarity_threshold:
                    self.G.add_edge(label1, label2, weight=self.cosine_matrix[i, j])

    def get_color_map(self):
        similarity_scores = np.mean(self.cosine_matrix, axis=1)
        norm_scores = (similarity_scores - np.min(similarity_scores)) / (np.max(similarity_scores) - np.min(similarity_scores))
        colors = [to_hex(cm.viridis(score)) for score in norm_scores]
        return colors

    def visualize_cluster_similarity_graph(self):
        partition = community_louvain.best_partition(self.G)
        self.communities = {node: partition[node] for node in self.G.nodes}

        unique_communities = list(set(self.communities.values()))
        color_palette = cm.get_cmap('tab20', len(unique_communities))
        community_colors = [to_hex(color_palette(i)) for i in range(len(unique_communities))]
        community_colors_map = {unique_communities[i]: community_colors[i] for i in range(len(unique_communities))}
        node_colors = [community_colors_map[self.communities[node]] for node in self.G.nodes]

        labels = {node: f"{self.G.nodes[node]['label']}:\n" + ", ".join(self.top_words[node]) for node in self.G.nodes}

        pos = nx.spring_layout(self.G)
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            text=[],
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_colors,
                size=10,
                line_width=2))

        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]

        for node in self.G.nodes():
            x, y = pos[node]
            node_trace['x'].append(x)
            node_trace['y'].append(y)
            node_trace['text'].append(labels[node])

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
        fig.update_layout(height=1200)
        fig.show()


def run(embeddings_path, df_path, col_name, n_dimensions=128, min_cluster_size=34, similarity_threshold=0.5):
    # Initialize ClusterVisualizer instance
    visualizer = ClusterVisualizer(embeddings_path, df_path)

    # Load embeddings
    visualizer.load_embeddings()

    # Cluster embeddings with specified parameters
    visualizer.cluster_embeddings(n_dimensions=n_dimensions, min_cluster_size=min_cluster_size)

    # Save cluster labels
    output_path = "../data dir/hdbscan_desc_labels.npy"
    visualizer.save_cluster_labels(output_path)

    # Load dataframe and specify column name
    visualizer.load_dataframe(col_name)

    # Get top TF-IDF words per cluster
    visualizer.get_top_tfidf_words_per_cluster(col_name)

    # Plot TSNE visualization of clusters
    visualizer.plot_tsne_clusters()

    # Build cluster similarity graph with the specified similarity threshold
    visualizer.build_cluster_graph(similarity_threshold)

    # Visualize cluster similarity graph
    visualizer.visualize_cluster_similarity_graph()


if __name__ == "__main__":
    # Example usage
    embeddings_path = input("Enter path to embeddings file: ")
    df_path = input("Enter path to dataframe file: ")
    col_name = input("Enter column name for clustering: ")
    min_cluster_size = int(input("Enter minimum cluster size: "))
    n_dimensions = int(input("Enter number of dimensions for embedding: "))
    similarity_threshold = float(input("Enter similarity threshold for clusters: "))  # Example threshold, you can adjust as needed

    run(embeddings_path, df_path, col_name, n_dimensions, min_cluster_size, similarity_threshold)
