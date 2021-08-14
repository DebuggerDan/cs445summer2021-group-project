import random
import math
import numpy as np

# works up to KCLUST = 8.  After that my plot data function breaks.
KCLUST = 10
INPUTS = 784
EPOCHS = 11
TRIALS = 10

# selects K number of points and returns as an array.
# chooses from col2 to start copying as it's 0-cluster, 1-labels
def randomPoints(array):
	size = np.shape(array)[0]
	means = np.zeros((KCLUST, INPUTS))
	for i in range(0, KCLUST):
		rando = random.randint(0, size)
		while array[rando][1] != i:
			rando = random.randint(0, size)
		means[i] = array[rando, 2:]

		# means[i] = array[random.randint(0, np.shape(array)[0]), 2:]
	return means

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


# recenters the mean to the center of it's cluster
def recenter_means(data, means):
	for i in range(0, KCLUST):
		count = 0
		total = np.zeros(INPUTS)
		for row in range(0, np.shape(data)[0]):
			if data[row][0] == i:
				count += 1
				total += data[row, 2:]
				
		# finds new position of center of cluster
		# unless is empty cluster, then stays the same
		if count != 0:
			means[i] = total / count

	return means 

# returns the sum of the L2 distance of clusters from means
def sum_of_squares(data, means):
	sumOsq = 0
	for i in range(0, np.shape(data)[0]):
		sumOsq += euclidean_dist(data[i, 2:], means[int(data[i][0])])

	return sumOsq

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

def accuracy(data):
	confusion = np.zeros((KCLUST, KCLUST), dtype=int)
	for row in range(0, np.shape(data)[0]):
		confusion[int(data[row][0])][int(data[row][1])] += 1
	correct, total = 0, 0
	for row in range(0, KCLUST):
		for col in range(0, KCLUST):
			total += confusion[row][col]
			if row == col:
				correct += confusion[row][col]
	return correct / total

def main():
	# creates data with labels.
	print("loading training data...")
	data = np.loadtxt("csv/mnist_train.csv", delimiter=",", skiprows=1)
	print("loading test data...")
	testdata = np.loadtxt("csv/mnist_test.csv", delimiter=",", skiprows=1)

	labels = np.zeros((np.shape(testdata)[0], 1))
	testdata = np.concatenate((labels, testdata), axis=1)

	labels = np.zeros((np.shape(data)[0], 1))
	data = np.concatenate((labels, data), axis=1)

	print("starting kmeans algorithm on training data...")
	smallest = math.inf
	biggest = 0
	# iterates a number of times prints our sum of squares each iteration
	for i in range(0, TRIALS):
		means = randomPoints(data)
		data = find_clusters(data, means)
		for j in range(0, EPOCHS):
			means = recenter_means(data, means)
			data = find_clusters(data, means)
		sqsum = sum_of_squares(data, means)
		print("Sum of Round", i+1, ":", sqsum)
		acc = accuracy(data)
		print("Accuracy: %{:.3f}".format(acc * 100))
		# records the smallest SoS we have gotten so far.
		if acc > biggest:
			# smallest = sqsum
			biggest = acc
			centroids = means	
	
	# print("These are our best points:\n", centroids)
	testdata = find_clusters(testdata, centroids)
	confusion_matty(testdata)

	# simple_plot(data, means, 10)

	

if __name__ == "__main__":
	main()