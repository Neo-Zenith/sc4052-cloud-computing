import json
import random
from typing import List, Dict
from multiprocessing import Pool
from Matrix import Matrix
import networkx as nx
import numpy as np
import math

class PageRank:
    def __init__(self, damping_factor: float, epsilon: float):
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        self.iteration = 0

    def calculate_page_rank(self, teleportation_bias: List[float], G: nx.DiGraph, map_reduce_mode: bool = False) -> List[float]:
        n = G.number_of_nodes()
        adjacency_matrix = nx.to_numpy_array(G)
        page_rank_scores = [1.0 / n] * n
        previous_page_rank_scores = [0.0] * n
        second_term = Matrix.multiply_scalar_vector(1 - self.damping_factor, teleportation_bias)

        while True:
            self.iteration += 1
            previous_page_rank_scores = page_rank_scores.copy()

            if map_reduce_mode:
                pool = Pool(processes=4)
                # Map the task to each process
                args = [(i, j, adjacency_matrix[i][j: j + n // 4], previous_page_rank_scores[j: j + n // 4]) for i in range(n) for j in range(0, n, n // 4)]
                results = pool.starmap(self._calculate_page_rank_element, args)
                pool.close()
                pool.join()

                # Reduce the results
                new_page_rank_scores = [0.0] * n
                for result in results:
                    i, j, value = result
                    new_page_rank_scores[i] += value
                for i in range(n):
                    new_page_rank_scores[i] = (1 - self.damping_factor) * teleportation_bias[i] + self.damping_factor * new_page_rank_scores[i]
                page_rank_scores = new_page_rank_scores

            else:
                sum_val = Matrix.multiply_matrix_vector(adjacency_matrix, previous_page_rank_scores)
                first_term = Matrix.multiply_scalar_vector(self.damping_factor, sum_val)
                page_rank_scores = Matrix.add_vector(first_term, second_term)

            if self._is_converged(page_rank_scores, previous_page_rank_scores):
                break

        return page_rank_scores

    def _calculate_page_rank_element(self, i, j, adjacency_matrix_elements, previous_page_rank_scores):
        return (i, j, sum(adjacency_matrix_elements[j] * previous_page_rank_scores[j] for j in range(len(adjacency_matrix_elements))))

    def calculate_distribution_vector(self, size: int, page_keywords_path: str, webpage_urls: List[str], allow_bias: bool, topic: str, bias_percentage: float) -> List[float]:
        page_keywords_map = self._load_page_keywords(page_keywords_path)
        teleportation_bias = [1] * size

        if allow_bias:
            for i, url in enumerate(webpage_urls):
                if url in page_keywords_map and topic in page_keywords_map[url]:
                    teleportation_bias[i] *= (1 + bias_percentage)

        _sum = sum(teleportation_bias)
        teleportation_bias = [bias / _sum for bias in teleportation_bias]

        return teleportation_bias

    def _is_converged(self, page_rank_scores: List[float], previous_page_rank_scores: List[float]) -> bool:
        return all(abs(page_rank_scores[i] - previous_page_rank_scores[i]) <= self.epsilon for i in range(len(page_rank_scores)))

    def _load_page_keywords(self, file_path: str) -> Dict[str, List[str]]:
        with open(file_path, 'r') as file:
            page_keywords_map = json.load(file)

        return page_keywords_map

    def load_webpage_urls(self, file_path: str, is_random: bool, size: int, seed: int) -> List[str]:
        with open(file_path, 'r') as file:
            urls = file.read().splitlines()

        if is_random:
            random.seed(seed)
            random.shuffle(urls)

        return urls[:size]

    def generate_sparse_graph(self, size: int, seed: int) -> nx.DiGraph:
        random.seed(seed)
        G = nx.DiGraph()
        G.add_nodes_from(range(size))
        # Add edges while limiting the out-degree to 10 and assigning random weights
        for i in range(size):
            out_degree = G.out_degree(i)
            edges = []
            out_degree_max = min(10, size)
            while out_degree < out_degree_max:
                # Select a random node to connect to
                j = random.randint(0, size - 1)
                # Check if the edge already exists
                if not G.has_edge(i, j):
                    # Assign a random weight to the edge
                    weight = random.random()
                    edges.append((j, weight))
                    out_degree += 1

            # Normalize the weights so they sum to 1
            sum_weights = sum(weight for _, weight in edges)
            for j, weight in edges:
                normalized_weight = weight / sum_weights
                G.add_edge(i, j, weight=normalized_weight)
        
        return G

    def generate_small_world_graph(self, size: int, seed: int) -> nx.DiGraph:
        # Generate a small-world graph using the Watts-Strogatz model
        G = nx.watts_strogatz_graph(size, k=4, p=0.3, seed=seed)
        # Create a directed graph
        DG = G.to_directed()

        # Generate a sample of the edges, then remove this sample from the graph
        edges = list(DG.edges())
        sample_size = int(len(edges) * 0.5)
        random.seed(seed)
        sample = random.sample(edges, sample_size)
        DG.remove_edges_from(sample)

        # Iterate over nodes to assign random edge weights and normalize them
        for node in DG.nodes():
            # Get outgoing edges
            out_edges = list(DG.out_edges(node))
            # Generate random weights for edges
            weights = np.random.rand(len(out_edges))
            # Normalize weights
            sum_weights = sum(weights)
            weights /= sum_weights
            # Assign weights to edges
            for i, edge in enumerate(out_edges):
                DG.edges[edge]['weight'] = weights[i]

        return DG
        
    def generate_random_scale_free_graph(self, size: int, seed: int) -> nx.DiGraph:
        # Generate a random scale-free graph using the Barabási–Albert model
        G = nx.barabasi_albert_graph(size, m=2, seed=seed)

        # Create a directed graph
        DG = G.to_directed()

        # Generate a sample of the edges, then remove this sample from the graph
        edges = list(DG.edges())
        sample_size = int(len(edges) * 0.5)
        random.seed(seed)
        sample = random.sample(edges, sample_size)
        DG.remove_edges_from(sample)

        # Iterate over nodes to assign random edge weights and normalize them
        for node in DG.nodes():
            # Get outgoing edges
            out_edges = list(DG.out_edges(node))
            # Generate random weights for edges
            weights = np.random.rand(len(out_edges))
            # Normalize weights
            sum_weights = sum(weights)
            weights /= sum_weights
            # Assign weights to edges
            for i, edge in enumerate(out_edges):
                DG.edges[edge]['weight'] = weights[i]

        return DG

    def generate_wiki_graph(self, seed: int, link_map_path: str, webpage_urls: List[str]) -> nx.DiGraph:
        random.seed(seed)

        with open(link_map_path, 'r') as file:
            link_map = json.load(file)

        G = nx.DiGraph()

        # Add nodes to the graph
        G.add_nodes_from(webpage_urls)

        for i, url in enumerate(webpage_urls):
            if url in link_map:
                links = link_map[url]
                for link in links:
                    if link in webpage_urls:
                        # Assign a random weight to the edge
                        weight = random.randint(1, len(webpage_urls))
                        G.add_edge(url, link, weight=weight)

        # Normalize the weights so they sum to 1 for each node
        for node in G.nodes():
            total_weight = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
            for neighbor in G.neighbors(node):
                G[node][neighbor]['weight'] /= total_weight

        return G

    def generate_sample_graph(self) -> nx.DiGraph:
        # Define the adjacency matrix
        adjacency_matrix = [
            [0, 0.5, 0, 0],
            [1/3, 0, 0, 0.5],
            [1/3, 0, 1, 0.5],
            [1/3, 1/2, 0, 0]
        ]
        
        # Create an empty directed graph
        G = nx.DiGraph()

        # Add nodes to the graph
        G.add_nodes_from(range(len(adjacency_matrix)))

        # Add edges to the graph based on the adjacency matrix
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix[i])):
                weight = adjacency_matrix[i][j]
                if weight != 0:
                    G.add_edge(i, j, weight=weight)

        return G

    def save_page_rank_scores(self, file_path: str, page_rank_scores: List[float]):
        with open(file_path, 'w') as file:
            for i, score in enumerate(page_rank_scores):
                file.write(f"{i},{score[0]}\n")
