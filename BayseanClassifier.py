import math


class BayeseanLearningModel:

	TRAIN_IMAGES = []
	TEST_IMAGES = []
		
	# practical infinities
	PINF = 1000000
	NINF = -1000000

	# prior class probabilities
	PRIOR_PROBS = []

	# counts of digit types in the training data
	COUNTS = []
	COUNTS_TOTAL = 0

	# training and test file descriptors
	TRAIN_FILE = None
	TEST_FILE = None

	# training and test data
	# 2D list of ints
	TRAIN_DATA = []
	TEST_DATA = []

	# Means / SDs for 0-9
	MEANS = []
	SD = []

	# confusion matrix
	confusion_matrix = []

	# constructor - takes trining and testing file descriptors
	# Two ways to use
	# 1) Generate new means and stdandard deviations with novel train and test data
	# 2) Use previously generated means and standard deviations
	def __init__(self, OPTION = None, file1 = None, file2 = None, file3 = None):
		
		if(OPTION == 'LOAD' and file1 != None and file2 != None and file3 != None):
			self.load_stats(file1, file2, file3)
		elif(OPTION == 'NEW' and file1 != None and file2 != None):
			self.read_files(file1, file2)
			self.init_prob_model() 
		else:
			print("BLM Constructor: <OPTION> <file1> <file2> <file3>")
			print("OPTION choices:")
			print("NEW")
			print("\tfile1 - training data file")
			print("\tfile2 - testing data file")
			print("\tfile3 - NONE - or leave blank")
			print("LOAD")
			print("\tfile1 - means data file")
			print("\tfile2 - standar deviations data file")
			print("\tfile3 - counts data file")
			exit(-1)

		self.compute_prior_probabilties()
		self.init_confusion_matrix()
	

	def read_files(self, train_file, test_file):
		self.TRAIN_FILE = train_file
		self.TEST_FILE = test_file
		self.read_train_data()
		self.read_test_data()


	# read training data into class data struture
	def read_train_data(self):
		with open(self.TRAIN_FILE, "r") as f:
			for line in f:
				row = []
				i = 0
				j = 1
				while(line[j] != '\n'):
					while(line[j] != '\n' and line[j] != ','):
						j += 1

					# read in the number and scale to be between 0 and 1
					num = float(line[i:j])
					if i > 0:
						num /= 255
					row.append(num)

					if(line[j] != '\n'):
						i = j+1
						j += 1
				
				self.TRAIN_DATA.append(row.copy())
				row.clear()
					
				num = float()
	

	# read testing data into class data struture
	def read_test_data(self):
		with open(self.TEST_FILE, "r") as f:
			for line in f:
				row = []
				i = 0
				j = 1
				while(line[j] != '\n'):
					while(line[j] != '\n' and line[j] != ','):
						j += 1

					# read in the number and scale to be between 0 and 1
					num = float(line[i:j])
					if i > 0:
						num /= 255
					row.append(num)

					if(line[j] != '\n'):
						i = j+1
						j += 1
				
				self.TEST_DATA.append(row.copy())
				row.clear()
					
				num = float()
	

	def load_stats(self, fM, fSD, fC):
		with open(fM, "r") as M:
			for line in M:
				row = []
				i = 0
				j = 1
				while(line[j] != '\n'):
					while(line[j] != '\n' and line[j] != ','):
						j += 1

					num = float(line[i:j])
					row.append(num)

					if(line[j] != '\n'):
						i = j+1
						j += 1
				
				self.MEANS.append(row.copy())
				row.clear()
					
				num = float()

		with open(fSD, "r") as SD:
			for line in SD:
				row = []
				i = 0
				j = 1
				while(line[j] != '\n'):
					while(line[j] != '\n' and line[j] != ','):
						j += 1

					num = float(line[i:j])
					row.append(num)

					if(line[j] != '\n'):
						i = j+1
						j += 1
				
				self.SD.append(row.copy())
				row.clear()
					
				num = float()

		with open(fC, "r") as C:
			for line in C:
				i = 0
				j = 1
				while(line[j] != '\n'):
					while(line[j] != '\n' and line[j] != ','):
						j += 1

					num = float(line[i:j])

					if(i == 0):
						self.COUNTS_TOTAL = int(num)
					else:
						self.COUNTS.append(int(num))


					if(line[j] != '\n'):
						i = j+1
						j += 1
				
					num = float()
	
		

		
	
	

	# initialize 10x10 confusion matrix
	def init_confusion_matrix(self):
		for i in range(10):
			row = []
			for j in range(10):
				row.append(0)
			self.confusion_matrix.append(row.copy())
			row.clear();


	# update confusion matrix at location [prediction, target_class]
	def update_confusion_matrix(self, p, c):
		#print(p, c)
		self.confusion_matrix[p][int(c)] += 1


	# calulate accuracy of model using confusion matrix
	def accuracy(self):
		total_correct = 0
		total = 0
		for i in range(10):
			for j in range(10):
				total += self.confusion_matrix[i][j]
				if i == j:
					total_correct += self.confusion_matrix[i][j]
		
		#print(total_correct,"/",total)
		return total_correct / total
	

	# calculate precision of all classes
	# returns array of percisions
	def precision(self):
		totals = [0 for i in range(10)]
		precisions = [0 for i in range(10)]
		for i in range(len(self.confusion_matrix)):
			for j in range(len(self.confusion_matrix[i])):
				totals[j] += self.confusion_matrix[i][j]
		
		for i in range(10):
			precisions[i] = self.confusion_matrix[i][i] / totals[i]
			precisions[i] = round(precisions[i], 3)
		
		return precisions
	
	def recall(self):
		totals = [0 for i in range(10)]
		recalls = [0 for i in range(10)]
		for i in range(10):
			for j in range(10):
				totals[i] += self.confusion_matrix[i][j]
		
		for i in range(10):
			recalls[i] = self.confusion_matrix[i][i] / totals[i]
			recalls[i] = round(recalls[i], 3)

		return recalls
	

	# Clears and resets Means and Standard Deviations
	def init_M_SD_PP(self):
		self.MEANS.clear()
		self.SD.clear()
		self.PRIOR_PROBS.clear()

		for i in range(10):
			self.PRIOR_PROBS.append(0)

			row = []
			for i in range(784): #default = 784
				row.append(0)
			self.MEANS.append(row.copy())
			self.SD.append(row.copy())
		

	# clears and resets counts and counts total
	def init_counts(self):
		self.COUNTS.clear()
		self.COUNTS_TOTAL = 0
		for i in range(10):
			self.COUNTS.append(0)


	def compute_prior_probabilties(self):
		for i in range(10):
			self.PRIOR_PROBS.append(self.COUNTS[i] / self.COUNTS_TOTAL)


	# initialzes the probabilistic model given training data
	# gets number of total/digits 0-9
	# calls funstions to calculate means and standard deviations
	def init_prob_model(self):
		self.init_counts()
		self.init_M_SD_PP()
		for line in self.TRAIN_DATA:
			# increments the count of of the given target
			self.COUNTS[int(line[0])] += 1
			self.COUNTS_TOTAL += 1

		self.compute_means()
		self.compute_SDs()
	

	# compute means of all parameters of each email given its class (img[-1])
	def compute_means(self):
		# accumulate summs for all means
		for img in self.TRAIN_DATA:
			target = int(img[0])

			# First element in img is the target. Target is not an attribute.
			# The length of self.MEANS is 1 less than length of an image.
			for i in range(1, len(img)):
				att = i-1
				self.MEANS[target][att] += img[i]
		
		# divide sums by related counts to aquire means
		for target in range(0, 10):
			for att in range(len(self.MEANS[target])):
				self.MEANS[target][att] /= self.COUNTS[target]


	# computes standard deviations of all parameters of each email given its class
	def compute_SDs(self):
		# sum (x-u)^2 for all parameters in each email givin its class
		for img in self.TRAIN_DATA:
			target = int(img[0])

			# first img element is the target. It is not an attribute
			for i in range(1, len(img)):
				att = i-1
				self.SD[target][att] += ((img[i] - self.MEANS[target][att])**2)

		# divide each SD sum by its class total and then take the square root	
		for target in range(0, 10):
			for att in range(0, 784):
				self.SD[target][att] /= self.COUNTS[target]
				self.SD[target][att] = math.sqrt(self.SD[target][att])
				
				#if self.SD[i][j] == 0:
				#	self.SD[i][j] = 0.0001


	# Niave Baysian Learning Model
	# Predicts class based on probabilistic model and test data	
	def testingGrounds(self, test_file):

		self.TEST_FILE = test_file
		self.read_test_data()

		# compute log(P(class)) for each class
		log_prior = []
		for i in range(10):
			log_prior.append( math.log10(self.PRIOR_PROBS[i]))

		# used to hold probabilities for each class for a given image
		c_prob = []

		# for all images
		for i in range(len(self.TEST_DATA)):
			# ensure start with 0 prob
			c_prob.clear()
			for k in range(10):
				c_prob.append(0)

			# find the prob of each class
			for cls in range(10):
			# add log(P(c)) to a prob
				c_prob[cls] += log_prior[cls]

				# for all elements in the email c_prob[cls] += SUM (log(N(index, x, class)))
				for j in range(784):
					c_prob[cls] += self.prob_xc(j, self.TEST_DATA[i][j+1], cls)
		

			prediction = 0
			for p in range(1, 10):
				if c_prob[p] > c_prob[prediction]:
					prediction = p


			# update confusion matrix with prediction and target class
			target = int(self.TEST_DATA[i][0])
			self.update_confusion_matrix(prediction, target)


	# Compute N(index, x, class) = (1/(sqrt(2pi)*sigma_ic)) * e^-((x-u)^2 / 2sigma_ic^2)
	def prob_xc(self, i, x, c):
		# check to see if SD is 0. if so everything is 0
		# avoids divide by 0
		if self.SD[c][i] == 0:
			return 0

		# calc 1 / (sqrt(2pi)*sd) in parts for easier debug
		j3 = self.SD[c][i]
		j2 = math.sqrt(2*math.pi)
		j = (1/(j2*j3))

		# exponent for e calcd in parts for easier debug
		# (x[i]-u[i])^2 / (2*SD[i]^2)
		numer = x
		numer -= self.MEANS[c][i]
		numer = numer**2

		denom = (self.SD[c][i]**2)
		denom *= 2

		k = numer / denom	

		# ensure e^-(exp)
		k *= -1

		# e^-k
		l = math.exp(k)

		# in case of 0 output to avoid log(0)
		if l == 0:
			return 0

		# log(a*e^-k)
		prob = math.log10(j * l)

		return prob