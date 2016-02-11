#!/usr/bin/env python
# encoding=utf-8

import logging

import numpy

logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)
TRACE = logging.DEBUG - 1


def load_dataset_from_file(filename):
    dataset = []
    labels = []
    num_features = None
    with open(filename) as infile:
        for line in infile:
            line = line.strip().split('\t')
            if num_features is None:
                num_features = len(line)
            dataset.append(list(map(float, line[:-1])))
            labels.append(float(line[-1]))
        return dataset, labels


def standarRegress(xArray, yArray):
    """使用普通最小二乘法求回归系数"""
    xMatrix = numpy.mat(xArray)
    yMatrix = numpy.mat(yArray).T
    xTx = xMatrix.T * xMatrix
    if numpy.linalg.det(xTx) == 0.0:
        logging.error('奇异矩阵无法求逆')
        return
    w = xTx.I * (xMatrix.T * yMatrix)
    # 或下面这个
    # w = numpy.linalg.solve(xTx, xMatrix.T * yMatrix)
    return w.A1


def lwlrRegress(testPoint, xArray, yArray, k=1.0):
    """局部加权线性回归(LWLR - Locally Weighted Linear Regression)
    给待预测点附近的每个点赋予一定的权重.
    LWLR使用"核"来对附近的点赋予更高的权重, 最常用的是高斯核.
    ===

    """
    xMatrix = numpy.mat(xArray)
    yMatrix = numpy.mat(yArray).T
    m, _n = xMatrix.shape
    # 利用高斯核初始化权重矩阵
    weights = numpy.mat(numpy.eye(m))
    for j in range(m):
        diffMat = testPoint - xMatrix[j, :]
        weights[j, j] = numpy.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMatrix.T * (weights * xMatrix)
    if numpy.linalg.det(xTx) == 0.0:
        logging.error('奇异矩阵无法求逆')
        return
    ws = xTx.I * (xMatrix.T * (weights * yMatrix))
    return testPoint * ws


def lwlrTest(testArray, xArray, yArray, k=1.0):
    """
    对于所有的测试点, 使用LWLR局部加权线性回归来计算预测值
    """
    m, _n = numpy.array(testArray).shape
    yHat = numpy.zeros(m)
    for i in range(m):
        yHat[i] = lwlrRegress(testArray[i], xArray, yArray, k)
    return yHat


def rssError(yArray, yHatArr):
    """计算预测误差"""
    yArray = numpy.array(yArray)
    yHatArr = numpy.array(yHatArr)
    return ((yArray - yHatArr)**2).sum()

"""缩减方法 -- 岭回归, 前向足部回归, lasso法"""


def ridgeRegress(xMatrix, yMatrix, lam=0.2):
    xTx = xMatrix.T * xMatrix
    _m, n = numpy.shape(xMatrix)
    denom = xTx + (numpy.eye(n) * lam)
    if numpy.linalg.det(denom) == 0.0:
        logging.error('奇异矩阵无法求逆')
        return
    ws = denom.I * (xMatrix.T * yMatrix)
    return ws


def ridgeTest(xArray, yArray):
    xMatrix = numpy.mat(xArray)
    yMatrix = numpy.mat(yArray).T
    # 标准化Y
    yMean = numpy.mean(yMatrix, 0)
    yMatrix = yMatrix - yMean     # to eliminate X0 take numpy.mean off of Y
    # 标准化X的每一维
    xMeans = numpy.mean(xMatrix, 0)   # calc numpy.mean then subtract it off
    xVar = numpy.var(xMatrix, 0)      # calc variance of Xi then divide by it
    xMatrix = (xMatrix - xMeans) / xVar

    numTestPts = 30
    _m, n = xMatrix.shape
    wMatrix = numpy.zeros((numTestPts, n))
    for i in range(numTestPts):
        ws = ridgeRegress(xMatrix, yMatrix, numpy.exp(i - 10))
        wMatrix[i, :] = ws.T
    return wMatrix


def main():
    Xs, Ys = load_dataset_from_file('ex0.txt')
    logging.info('原始数据\n{0}'.format([(x, y) for x, y in zip(Xs, Ys)]))

    w = standarRegress(Xs, Ys)
    logging.info('最小二乘法回归系数: {0}'.format(w))
    logging.info('预测序列\n{0}'.format([
        (x, y) for x, y in map(
            lambda x: (x, float(numpy.mat(x) * numpy.mat(w).T)),
            Xs
        )
    ]))

    # k = 0.003
    k = 0.01
    # k = 0.1
    yHat = lwlrTest(Xs, Xs, Ys, k=k)
    logging.info('LWLR预测序列, 系数 k={0}\n{1}'.format(k, [
        (x, y) for x, y in zip(Xs, yHat)
    ]))

    '''
    # 绘制图看拟合效果
    xMatrix = numpy.mat(Xs)
    sorted_index = xMatrix[:, 1].argsort(axis=0)
    xSort = xMatrix[sorted_index][:, 0, :]
    import matplotlib.pyplot as plt
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[sorted_index])  # 拟合曲线
    ax.scatter(
        xMatrix[:, 1].A1, numpy.mat(Ys).T.A1,
        s=2, c='red'
    )  # 原始数据
    plt.show()
    '''

    abaloneXs, abalineYs = load_dataset_from_file('abalone.txt')
    w = ridgeTest(abaloneXs, abalineYs)


if __name__ == '__main__':
    main()


def regularize(xMatrix):#regularize by columns
    inMat = xMatrix.copy()
    inMeans = numpy.mean(inMat,0)   #calc numpy.mean then subtract it off
    inVar = numpy.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArray,yArray,eps=0.01,numIt=100):
    xMatrix = numpy.mat(xArray); yMatrix=numpy.mat(yArray).T
    yMean = numpy.mean(yMatrix,0)
    yMatrix = yMatrix - yMean     #can also regularize ys but will get smaller coef
    xMatrix = regularize(xMatrix)
    m,n=numpy.shape(xMatrix)
    #returnMat = numpy.zeros((numIt,n)) #testing code remove
    ws = numpy.zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = numpy.inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMatrix*wsTest
                rssE = rssError(yMatrix.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        #returnMat[i,:]=ws.T
    #return returnMat

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()
    
from time import sleep
import json
import urllib2
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
    
def crossValidation(xArray,yArray,numVal=10):
    m = len(yArray)                           
    indexList = range(m)
    errorMat = numpy.zeros((numVal,30))#create error numpy.mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        numpy.random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArray[indexList[j]])
                trainY.append(yArray[indexList[j]])
            else:
                testX.append(xArray[indexList[j]])
                testY.append(yArray[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = numpy.mat(testX); matTrainX=numpy.mat(trainX)
            meanTrain = numpy.mean(matTrainX,0)
            varTrain = numpy.var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * numpy.mat(wMat[k,:]).T + numpy.mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,numpy.array(testY))
            #print errorMat[i,k]
    meanErrors = numpy.mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[numpy.nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/numpy.var(x)
    #we can now write in terms of x not Xreg:  x*w/numpy.var(x) - meanX/numpy.var(x) +meanY
    xMatrix = numpy.mat(xArray); yMatrix=numpy.mat(yArray).T
    meanX = numpy.mean(xMatrix,0); varX = numpy.var(xMatrix,0)
    unReg = bestWeights/varX
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(numpy.multiply(meanX,unReg)) + numpy.mean(yMatrix)