'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
import numpy

def loadDataset(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def binSplitDataset(dataset, feature, value):
    mat0 = dataset[numpy.nonzero(dataset[:,feature] > value)[0],:][0]
    mat1 = dataset[numpy.nonzero(dataset[:,feature] <= value)[0],:][0]
    return mat0,mat1

def regLeaf(dataset):#returns the value used for each leaf
    return numpy.mean(dataset[:,-1])

def regErr(dataset):
    return numpy.var(dataset[:,-1]) * numpy.shape(dataset)[0]

def linearSolve(dataset):   #helper function used in two places
    m,n = numpy.shape(dataset)
    X = numpy.mat(numpy.ones((m,n)))
    Y = numpy.mat(numpy.ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataset[:,0:n-1]; Y = dataset[:,-1]#and strip out Y
    xTx = X.T*X
    if numpy.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataset):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataset)
    return ws

def modelErr(dataset):
    ws,X,Y = linearSolve(dataset)
    yHat = X * ws
    return sum(numpy.power(Y - yHat,2))

def chooseBestSplit(dataset, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataset[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataset)
    m,n = numpy.shape(dataset)
    #the choice of the best feature is driven by Reduction in RSS error from numpy.mean
    S = errType(dataset)
    bestS = numpy.inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataset[:,featIndex]):
            mat0, mat1 = binSplitDataset(dataset, featIndex, splitVal)
            if (numpy.shape(mat0)[0] < tolN) or (numpy.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataset) #exit cond 2
    mat0, mat1 = binSplitDataset(dataset, bestIndex, bestValue)
    if (numpy.shape(mat0)[0] < tolN) or (numpy.shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataset)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataset, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataset is NumPy numpy.mat so we can array filtering
    feat, val = chooseBestSplit(dataset, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataset(dataset, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if numpy.shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataset(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataset(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(numpy.power(lSet[:,-1] - tree['left'],2)) +\
            sum(numpy.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(numpy.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = numpy.shape(inDat)[1]
    X = numpy.mat(numpy.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = numpy.mat(numpy.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, numpy.mat(testData[i]), modelEval)
    return yHat