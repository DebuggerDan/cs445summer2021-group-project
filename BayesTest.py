import BayseanClassifier as bc
# files for LOAD option
fileM = "Means.txt"
fileSD = "Standard_Deviations.txt"
fileC = "Counts.txt"

# files for NEW option
fileTrain = "mnist_train.csv"
fileTest = "mnist_test.csv"


# Creates Bayes Model Classifier
# 2 Options: NEW or LOAD
b = bc.BayeseanLearningModel("LOAD", fileM, fileSD, fileC)

# After Bayes model is created or loaded (aka trained) run tests
b.testingGrounds(fileTest)

# Print resulting Accuracy, Precisions, and Recalls
print("Accuracy:", b.accuracy())
prec = b.precision()
rec = b.recall()
avg_prec = 0
avg_rec = 0
for i in range(10):
	avg_prec += prec[i]
	avg_rec += rec[i]
avg_prec /= 10
avg_rec /= 10
print("Average Precision: ", avg_prec)
print("Average Recall: ", avg_rec)
print("Precision Per Digit: ", b.precision())
print("Recall Per Digit: ", b.recall())
for i in range(10):
	print(b.confusion_matrix[i])
