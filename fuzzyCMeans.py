# hopefully an algo to solve MNIST data using the fuzzy c means
# algorithm in some capacity.

import numpy as np
import matplotlib.pyplot as plt

CCLUSTERS = 10
FUZZIFIER = 3
# larger fuzzifier is the more classes are 'washed out'

def init_weights(data):
	size = np.shape(data)[0]
	print("size is", size)

	weights = np.random.rand(size, CCLUSTERS)
	print("size of weights:", np.shape(weights))
	print("First 10 weight rows:\n", weights[:10, :])

	return weights

def compute_centroid(weights, data):
	print("This is where we'll compute the centroids...")

	centroids = np.zeros((CCLUSTERS, np.shape(data)[1] - 1))
	for cluster in range(0, CCLUSTERS):
		fuzzy_weights = np.power(weights[:, cluster], FUZZIFIER)
		clustSum = np.sum(fuzzy_weights)
		print("Cluster:", cluster, "Bottom Sum:", clustSum, "fuzzy size:", np.shape(fuzzy_weights))
		# do from 1 to avoid the labels...
		for col in range(1, np.shape(data)[1]):
			centroids[cluster][col - 1] = np.dot(fuzzy_weights, data[:, col]) / clustSum

	print("the first centroid:\n", centroids[0, :])
	print("size of centroids: ", np.shape(centroids))
	return centroids

			
		

def main():
	print("Reading in Test Data...")
	test_data = np.loadtxt("csv/mnist_test.csv", delimiter=",", skiprows=1)
	print("Size of test data:", np.shape(test_data))

	centroids = np.zeros((CCLUSTERS, np.shape(test_data)[1] - 1))
	print("size of centroids:", np.shape(centroids))
	weights = init_weights(test_data)

	centroids = compute_centroid(weights, test_data)


if __name__ == "__main__":
	main()