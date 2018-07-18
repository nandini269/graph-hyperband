import sys
import numpy as np 
import csv

def createtv(filename):

	r = csv.reader(open(filename), delimiter=",")
	res = np.array(list(r))
	head = res[0][0]
	for i in range(1,len(res[0])):
		head = head + ',' + res[0][i]
	#print(head)
	res = res[1:]
	l = len(res)
	# random seed
	np.random.seed(279)
	np.random.shuffle(res)
	sep = int(round(0.65 * l))
	train = res[:sep]
	sep2 = int(round((l - sep)/2))
	valid = res[sep:sep+sep2]
	test = res[sep+sep2:]
	np.savetxt("train.csv", train, fmt='%s', delimiter=',', header= head)
	np.savetxt("valid.csv", valid, fmt='%s', delimiter=',', header= head)
	np.savetxt("test.csv", test, fmt='%s', delimiter=',', header= head)

filename = sys.argv[1]
createtv(filename)
