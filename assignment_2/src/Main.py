import os
from PageRank import PageRank
from Matrix import Matrix
import networkx as nx

def main():
    damping_factor = 0.65
    random_seed = 19
    is_random_url = True    ## Only set to False when adjusting E
    allow_bias = False      ## Only set to True when adjusting E
    topic_of_interest = 'article'
    bias_percentage = 0.2
    epsilon = 1e-6
    n = 100
    map_reduce_mode = False

    pageRank = PageRank(damping_factor, epsilon)

    # Load webpage URLs from the txt file
    webpageUrls = pageRank.load_webpage_urls(os.path.abspath("../output/urls.txt"), is_random_url, n, random_seed)
    # Compute teleportation vector
    teleportation_vector = pageRank.calculate_distribution_vector(len(webpageUrls), "../output/top_keywords.json", webpageUrls, allow_bias, topic_of_interest, bias_percentage)
    
    # Build the graph
    graph = pageRank.generate_wiki_graph(random_seed, os.path.abspath("../output/links.json"), webpageUrls)
    # graph = pageRank.generate_sparse_graph(len(webpageUrls), random_seed)
    # graph = pageRank.generate_random_scale_free_graph(len(webpageUrls), random_seed)
    # graph = pageRank.generate_small_world_graph(len(webpageUrls), random_seed)
    # graph = pageRank.generate_sample_graph()  # Used only for n = 4

    power_method_scores = power_method(pageRank, teleportation_vector, graph, map_reduce_mode)
    invert_matrix_method_scores = invert_matrix_method(damping_factor, teleportation_vector, graph)
    sanity_check(power_method_scores, invert_matrix_method_scores)

    # Save the PageRank scores to a txt file
    file_name = f"../output/ranks/pagerank_scores_n={n}_"
    if allow_bias:
        file_name += f"bias_{bias_percentage * 100:.0f}"
    else:
        file_name += "no_bias"
    file_name += ".txt"
    pageRank.save_page_rank_scores(file_name, power_method_scores)
        

def power_method(pageRank, teleportation_vector, adjacency_matrix, map_reduce_mode):
    print("=== Power Method ===")
    # Calculate the PageRank scores
    page_rank_scores = pageRank.calculate_page_rank(teleportation_vector, adjacency_matrix, map_reduce_mode)

    # Print the PageRank scores
    print("PageRank algorithm converged.")
    # for i, score in enumerate(page_rank_scores):
    #     print(f"Page {i + 1}: {score:.4f}")
    print(f"Iterations taken to converge: {pageRank.iteration}")
    print("=== End of Power Method ===")
    return page_rank_scores

def invert_matrix_method(damping_factor, teleportation_vector, G: nx.DiGraph):
    print("=== Invert Matrix Method ===")
    # Formula is (I - dampingFactor * M)^-1 * (1 - dampingFactor) * teleportationVector
    adjacency_matrix = nx.to_numpy_array(G)
    identity_matrix = Matrix.create_identity_matrix(len(adjacency_matrix))
    first_term = Matrix.subtract_matrix(identity_matrix, Matrix.multiply_scalar_matrix(damping_factor, adjacency_matrix))
    inverted_matrix = Matrix.invert_matrix(first_term)
    second_term = Matrix.multiply_scalar_vector(1 - damping_factor, teleportation_vector)
    page_rank_scores = Matrix.multiply_matrix_vector(inverted_matrix, second_term)

    # Print the PageRank scores
    print("PageRank algorithm converged.")
    # for i, score in enumerate(page_rank_scores):
    #     print(f"Page {i + 1}: {score[0]:.4f}")
    print("=== End of Invert Matrix Method ===")

    return page_rank_scores

def sanity_check(power_method_scores, invert_matrix_scores):
    for i in range(len(power_method_scores)):
        if abs(power_method_scores[i] - invert_matrix_scores[i]) > 1e-3:
            print("The PageRank scores from the Power Method and Invert Matrix Method are not equal.")
            return
    print("The PageRank scores from the Power Method and Invert Matrix Method are equal.")

def recordPerformance(file_path: str, n: int, duration: float):
    with open(file_path, 'a') as file:
        file.write(f"{n}, {duration:.4f}\n")

if __name__ == "__main__":
    main()
