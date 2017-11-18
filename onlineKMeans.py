#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:38:26 2017

@author: tengpan
"""
import numpy as np
from numpy import array
import gzip
from numpy import array
from numpy.linalg import norm
import random

#--------------------------------Analysis--------------------------------------
##read into a matrix
def zp2matrix():
    matrix = []
    count = 0 #about 4,600,000 line
    with gzip.open('ydata-fp-td-clicks-v1_0.20090501.gz','r') as f:
        for line in f:
            count += 1
            #get user
            user = line.decode("utf-8").split("|")[1]
            rate = user.split(" ")[1:7]
            res = [x[2:] for x in rate]
            #set feature at beginning
            res.insert(0, res.pop())
            #float type
            res = np.asfarray(res,float)
            matrix.append(res)
            if (count % 100000 == 0):
                print (count)
    return matrix, count


##randomized centroid initialization
def initRandom(matrix, k):
    return random.sample(matrix, k)
    #return np.vstack(random.sample(matrix, 3))
    #np.matrix(s)


##k-means++ initialization
def initKMeans(matrix, k):
    #Sample first center uniformly 
    matrixCenter = random.sample(matrix, 1)
    candiCenter = matrixCenter[0] 
    #Compute proposal distribution q(x) if assume not uniform
    #Sequentially construct k-1 independent Markov chains -1=00 to obtain k-1 
    #cluster centers
    for i in range (2, k + 1):
        matrixSample = random.sample(matrix, 100)
        candiCenter = nextCenter(matrixSample, matrixCenter)
        matrixCenter.append(candiCenter)
    return matrixCenter
    #return np.vstack(random.sample(matrix, 3))
    
def nextCenter(matrixSample, matrixCenter):
    candiCenter = matrixSample[0]
    candiCenterD = minDistance(matrixCenter, candiCenter)
    for sample in matrixSample[1:]:
        #sampleD, indexD = minDistance(matrixCenter, sample)
        sampleD = minDistance(matrixCenter, sample)
        if (candiCenterD == 0 or sampleD >= candiCenterD):
            p = 1
        else:
            p = sampleD / candiCenterD
        rand = random.uniform(0, 1)
        if (rand <= p):
            candiCenter = sample
            candiCenterD = sampleD
    return candiCenter
def minDistance(matrixCenter, sample):
    d = np.linalg.norm(sample - matrixCenter[0])
    #count = 0;
    #index = 0;
    for center in matrixCenter[1:]:
        #count += 1
        #distance
        dnew = np.linalg.norm(sample - center)
        if (dnew < d):
            d = dnew
            #index = count
    return pow(d, 2)#, index
    #return pow(d, 2), index


##Mini-batch k-means
def onlineKMeans(matrixCenter, matrix, T):
    #list of array
    #matrixCenter = np.matrix(matrixCenter)
    #minBath
    B = 20
    for t in range(1, T+1):      
        matrixSample = random.sample(matrix, B)
        matrixCenter = updateDeltaCenter(matrixCenter, matrixSample, t)
    return matrixCenter

def updateDeltaCenter(matrixCenter, matrixSample, t):
    matrixUpdate = matrixCenter
    n = 1 / t
    for sample in matrixSample:
        dis2Centers = array([norm(v) for v in matrixCenter - sample])
        minIndex = np.argmin(dis2Centers)
        matrixUpdate[minIndex] += n * (sample - matrixCenter[minIndex])
    return matrixUpdate
                           
        



#-----------------------------main function------------------------------------
if __name__ == '__main__':
    print ('Hello word')
    #f = gzip.GzipFile('ydata-fp-td-clicks-v1_0.20090501.gz', "r")
    matrix, count = zp2matrix()
    