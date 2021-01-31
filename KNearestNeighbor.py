import numpy as np
import operator


# function to calculate euclidean distance
def euc_dist(x1, x2):
	return np.sqrt(np.sum((x1 - x2) ** 2))


class KNearestNeighbor():
	
	def __init__(self, K):
		self.K = K
	
	def fit(self, x_train, y_train):
		self.X_train = x_train
		self.Y_train = y_train
	
	def predict(self, X_test):
		
		# list to store all our predictions
		predictions = []
		
		# loop over all observations
		for i in range(len(X_test)):
			
			# calculate the distance between the test point and all other points in the training set
			dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
			
			# sort the distances and return the indices of K neighbors
			dist_sorted = dist.argsort()[:self.K]
			
			# get the neighbors
			neigh_count = {}
			
			# for each neighbor find the class
			for idx in dist_sorted:
				if self.Y_train[idx] in neigh_count:
					neigh_count[self.Y_train[idx]] += 1
				else:
					neigh_count[self.Y_train[idx]] = 1
			
			# get the most common class label
			sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
			
			predictions.append(sorted_neigh_count[0][0])
		return predictions