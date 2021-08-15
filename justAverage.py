import random
import math
import numpy as np

# works up to KCLUST = 8.  After that my plot data function breaks.
KCLUST = 10
INPUTS = 784
EPOCHS = 9
TRIALS = 10
TEST_LOC = "csv/mnist_test.csv"
TRAIN_LOC = "csv/mnist_train.csv"

def confusion_matty(data):
	confusion = np.zeros((KCLUST, KCLUST), dtype=int)

	for row in range(0, np.shape(data)[0]):
		confusion[int(data[row][0])][int(data[row][1])] += 1

	print("Confusion Matrix, rows=guesses, cols=truths:\n", confusion)
	correct, total = 0, 0
	for row in range(0, KCLUST):
		for col in range(0, KCLUST):
			total += confusion[row][col]
			if row == col:
				correct += confusion[row][col]
	print("Accuracy:", correct, "/", total)
	print("Percentage: ", correct / total)

def euclidean_dist(ptA, ptB):
	subtracted = np.subtract(ptA, ptB)
	sumOsquares = np.dot(subtracted, subtracted)
	return math.sqrt(sumOsquares)

# finds the points with the smalles L2 distance from means and labels them
# as belongs to the cluster around said means.
def find_clusters(data, centroids):
	num_pts = np.shape(data)[0]
	for i in range(0, num_pts):
		smallest = math.inf
		for k in range(0, KCLUST):
			# finds all the L2 distances from point to centers
			l2 = euclidean_dist(data[i, 2:], centroids[k, :])
			# selects the smallest Euclidean (L2) distance.
			if l2 < smallest: 
				smallest = l2
				loc = k
		data[i][0] = loc
	
	# returns updated cluster / point data
	return data

def find_averages(array):
	centroids = np.zeros((KCLUST, INPUTS + 1))
	for row in range(0, np.shape(array)[0]):
		centroids[int(array[row][0]), 1:] += array[row, 1:]
		centroids[int(array[row][0])][0] += 1

	for row in range(0, KCLUST):
		centroids[row] = centroids[row] / centroids[row][0]

	return centroids[:, 1:]

def main():
	# creates data with labels.
	print("loading training data...")
	data = np.loadtxt(TRAIN_LOC, delimiter=",", skiprows=1)
	print("loading test data...")
	testdata = np.loadtxt(TEST_LOC, delimiter=",", skiprows=1)
	labels = np.zeros((np.shape(testdata)[0], 1))
	testdata = np.concatenate((labels, testdata), axis=1)

	print("finding averages of all labeled training data...")
	centroids = find_averages(data)

	# runs the clustering on the testdata from best centroids.
	print("assigning clusters to test data...")
	testdata = find_clusters(testdata, centroids)
	confusion_matty(testdata)
	



if __name__ == "__main__":
	main()