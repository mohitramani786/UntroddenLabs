import numpy
# from sklearn.cross_validation import train_test_split
import operator
import sys
import time
import timeit
import profile

def getNeighbors(trainingSet, testInstance, k):
	distances=[]
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append([trainingSet[x], dist])
	distances=numpy.array(distances)
	distances[numpy.argsort(distances[:,1])]
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 


def euclideanDistance(instance1,instance2,length):
	a=numpy.array(instance1[0:-1])
	b=numpy.array(instance2[0:-1])
	
	squarer = lambda t: float(t) 
	a = numpy.array([squarer(xi) for xi in a])
	b = numpy.array([squarer(xi) for xi in b])
	return numpy.linalg.norm(a-b)


def getAccuracy(testSet, predictions):
	correct = 0
	length=len(testSet)
	for x in range(length):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return numpy.multiply(numpy.divide(correct,float(length)),100.0)


def main():

	with open("iris.data", "r") as f:
		data = f.read().split('\n')
		data = numpy.array(data)  #convert array to numpy type array
		x_train ,x_test =(data[0:67],data[68:])
		# x_train ,x_test = train_test_split(data,test_size=0.67)
		print (data)
		x_train=[i for i in numpy.core.defchararray.split(x_train, sep=',')]
		x_test=[i for i in numpy.core.defchararray.split(x_test, sep=',')]
		# print('Train set: ' + repr(len(x_train)))
		# print('Test set: ' + repr(len(x_test)))
		predictions=[]
		k = 3	
		for x in range(len(x_test)):
			neighbors = getNeighbors(x_train, x_test[x], k)
			result = getResponse(numpy.array(neighbors))
			predictions.append(result)
			# print('> predicted=' + repr(result) + ', actual=' + repr(x_test[x][-1]))
			
		accuracy = getAccuracy(x_test, predictions)
		# print('Accuracy: ' + repr(accuracy) + '%')


main()