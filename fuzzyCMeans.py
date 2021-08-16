# hopefully an algo to solve MNIST data using the fuzzy c means
# algorithm in some capacity.

import numpy as np
import matplotlib.pyplot as plt
import math

CCLUSTERS = 10
FUZZIFIER = 9
# larger fuzzifier is the more classes are 'washed out'
EPOCHS = 10

# initialized all the weights to random amounts btwn 0-1
def init_weights(data):
	print("Initializing random weights...")
	size = np.shape(data)[0]
	weights = np.random.rand(size, CCLUSTERS)
	return weights

# computes the centroid of data points based off of weights and data
def compute_centroid(weights, data):
	print("computing the centroids...")

	centroids = np.zeros((CCLUSTERS, np.shape(data)[1] - 1))
	for cluster in range(0, CCLUSTERS):
		fuzzy_weights = np.power(weights[:, cluster], FUZZIFIER)
		clustSum = np.sum(fuzzy_weights)
		# do from 1 to avoid the labels...
		for col in range(1, np.shape(data)[1]):
			centroids[cluster][col - 1] = np.dot(fuzzy_weights, data[:, col]) / clustSum
	return centroids

# uses numpy to find euclidean distance between two points.
def euclidean_dist(ptA, ptB):
	subtracted = np.subtract(ptA, ptB)
	sumOsquares = np.dot(subtracted, subtracted)
	return math.sqrt(sumOsquares)

# computes the weights for the next round
def compute_weights(weights, data, centroids):
	print("recomputing weights...")
	exponent = 2 / (FUZZIFIER - 1)
	(rows, cols) = np.shape(weights)

	# temp holder will keep all the euclid dist.
	eu_dist = np.zeros((rows, cols))

	for row in range(0, rows):
		for col in range(0, cols):
			eu_dist[row][col] = euclidean_dist(data[row, 1:], centroids[col, :])

	for row in range(0, rows):
		for col in range(0, cols):
			sum = 0
			for secCol in range(0, cols):
				sum += math.pow(eu_dist[row][col] / eu_dist[row][secCol], exponent)
			weights[row][col] = 1 / sum

	return weights
			
# determines the sum of the weights multiplied by euclidean dist to data
def objective_func(weights, data, centroids):
	sum = 0
	(rows, cols) = np.shape(weights)

	for row in range(0, rows):
		for col in range(0, cols):
			sum += weights[row][col] * euclidean_dist(data[row, 1:], centroids[col, :])

	return sum

# creates a confusion matrix for the data
# BUG IT DOES NOT HOWEVER HAVE ANY ASSOCIATION WITH CENTROIDS
# TO ANY CERTAIN NUMBER SO ACC CAN NOT BE KNOWN.
def confusion_matty(weights, data, centroids):
	confusion = np.zeros((10, 10), dtype=int)
	rows = np.shape(data)[0]
	cols = np.shape(centroids)[0]

	# chooses the c means with the biggest weight for each of the data
	# points and assigns that for the centroid
	for row in range(0, rows):
		biggest = np.max(weights[row])
		for col in range(0, cols):
			if biggest == weights[row][col]:
				confusion[col][int(data[row][0])] += 1

	print("Confusion Matrix, rows=guesses, cols=truths:\n", confusion)

	correct, total = 0, 0
	for row in range(0, 10):
		for col in range(0, 10):
			total += confusion[row][col]
			if row == col:
				correct += confusion[row][col]

	print("Accuracy:", correct, "/", total)
	print("Percentage: ", correct / total)

def main():
	print("Reading in Test Data...")
	test_data = np.loadtxt("csv/mnist_test.csv", delimiter=",", skiprows=1)
	print("Size of test data:", np.shape(test_data))

	centroids = np.zeros((CCLUSTERS, np.shape(test_data)[1] - 1))
	print("size of centroids:", np.shape(centroids))
	weights = init_weights(test_data)
	objfun = objective_func(weights, test_data, centroids)
	print("initial objective func is:", objfun)

	# performs the M and the E step for a certain number of epochs.
	for i in range(0, EPOCHS):
		print("iteration {}...".format(i + 1))
		centroids = compute_centroid(weights, test_data)
		weights = compute_weights(weights, test_data, centroids)
		objfun = objective_func(weights, test_data, centroids)
		print("Our minimizing objective func is:", objfun)

	# RESULTS ARE NOT LISTED TO PARTICULAR DATA PTS...
	confusion_matty(weights, test_data, centroids)
	print("Centroids:", centroids)


if __name__ == "__main__":
	main()