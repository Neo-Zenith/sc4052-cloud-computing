import matplotlib.pyplot as plt

with open('../output/ranks/pagerank_scores_n=100_no_bias.txt', 'r') as file:
    ## Read the file
    pagerank_scores_unbiased = file.readlines()
    ## Remove the newline character from each line, split the line by the , character, and store the last element in a list
    pagerank_scores_unbiased = [float(line.strip().split(',')[-1]) for line in pagerank_scores_unbiased]

with open('../output/ranks/pagerank_scores_n=100_bias_80.txt', 'r') as file:
    ## Read the file
    pagerank_scores_bias80 = file.readlines()
    ## Remove the newline character from each line, split the line by the , character, and store the last element in a list
    pagerank_scores_bias80 = [(float(line.strip().split(',')[-1]) - pagerank_scores_unbiased[i])/ pagerank_scores_unbiased[i] for i, line in enumerate(pagerank_scores_bias80)]

with open('../output/ranks/pagerank_scores_n=100_bias_50.txt', 'r') as file:
    ## Read the file
    pagerank_scores_bias50 = file.readlines()
    ## Remove the newline character from each line, split the line by the , character, and store the last element in a list
    pagerank_scores_bias50 = [(float(line.strip().split(',')[-1]) - pagerank_scores_unbiased[i]) / pagerank_scores_unbiased[i] for i, line in enumerate(pagerank_scores_bias50)]

with open('../output/ranks/pagerank_scores_n=100_bias_20.txt', 'r') as file:
    ## Read the file
    pagerank_scores_bias20 = file.readlines()
    ## Remove the newline character from each line, split the line by the , character, and store the last element in a list
    pagerank_scores_bias20 = [(float(line.strip().split(',')[-1]) - pagerank_scores_unbiased[i]) / pagerank_scores_unbiased[i] for i, line in enumerate(pagerank_scores_bias20)]


## Plot a scatter plot of all 3 lists
plt.scatter(range(len(pagerank_scores_bias80)), pagerank_scores_bias80, label='Bias 80%')
plt.scatter(range(len(pagerank_scores_bias50)), pagerank_scores_bias50, label='Bias 50%')
plt.scatter(range(len(pagerank_scores_bias20)), pagerank_scores_bias20, label='Bias 20%')

## Plot a horizontal line at y=0 (unbias line)
plt.axhline(y=0, color='red', linestyle='--', label='Unbiased Line')

## Add a title to the plot
plt.title('Relative Difference in Biased PageRank Scores To Unbiased PageRank Scores')
## Add a label to the x-axis
plt.xlabel('Webpage')
## Add a label to the y-axis
plt.ylabel('PageRank Score Relative Difference')
## Add a legend to the plot
plt.legend()
## Show the plot
plt.show()