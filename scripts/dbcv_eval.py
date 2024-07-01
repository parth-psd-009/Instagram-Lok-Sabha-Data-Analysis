import numpy as np
from hdbscan import HDBSCAN
import hdbscan
import umap
import dbcv
import logging

class ClusterEvaluator:
    def __init__(self, embeddings_path, n_components=10, min_cluster_sizes=range(5, 100, 5)):
        self.embeddings_path = embeddings_path
        self.n_components = n_components
        self.min_cluster_sizes = min_cluster_sizes
        self.embeddings = None
        self.reduced_embeddings = None
        self.dbcv_scores = []
        self.cluster_counts = []

        # Initialize logging
        logging.basicConfig(level=logging.INFO)

    def load_embeddings(self):
        self.embeddings = np.load(self.embeddings_path)

    def reduce_dimensionality(self):
        reducer = umap.UMAP(n_components=self.n_components, random_state=42)
        self.reduced_embeddings = reducer.fit_transform(self.embeddings)

    def evaluate_dbcv(self):
        for size in self.min_cluster_sizes:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
                labels = clusterer.fit_predict(self.reduced_embeddings)

                if len(set(labels)) > 1:  # Ensure there is more than one cluster
                    dbcv_score = dbcv.dbcv(self.reduced_embeddings, labels, noise_id=-1)
                else:
                    dbcv_score = -1  # Invalid DBCV score if only one cluster

                self.dbcv_scores.append(dbcv_score)
                self.cluster_counts.append(len(set(labels)) - (1 if -1 in labels else 0))  # Exclude noise

            except MemoryError as e:
                logging.error(f"MemoryError at min_cluster_size={size}: {e}")
                self.dbcv_scores.append(None)  # Indicate failure
                self.cluster_counts.append(0)

    def filter_scores(self):
        filtered_dbcv_scores = [score for score in self.dbcv_scores if score is not None]
        filtered_cluster_counts = [count for i, count in enumerate(self.cluster_counts) if self.dbcv_scores[i] is not None]
        return filtered_dbcv_scores, filtered_cluster_counts

    def log_results(self):
        filtered_dbcv_scores, filtered_cluster_counts = self.filter_scores()
        logging.info(f"DBCV Scores: {filtered_dbcv_scores}")
        logging.info(f"Cluster Counts: {filtered_cluster_counts}")

    def run(self):
        self.load_embeddings()
        self.reduce_dimensionality()
        self.evaluate_dbcv()
        self.log_results()

# Usage
evaluator = ClusterEvaluator(embeddings_path="../data dir/desc_embeddings.npy")
evaluator.run()
