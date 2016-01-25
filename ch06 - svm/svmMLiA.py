#!/usr/bin/env python
# encoding=utf-8

"""
SVM - 支持向量机
===
介绍的是SVM的其中一种实现 -- 序列最小化(SMO, Sequential Minimal Optimization)算法
`分隔超平面` -- 将数据集分隔开来的超平面, 也就是分类的决策边界.
`间隔` -- 找到离分隔超平面最近的点, 确保他们离分隔面的距离尽可能远, 这其中点到分隔面的距离就是间隔.
    我们希望间隔尽可能地大, 以保证分类器尽可能健壮
`支持向量` -- 离分隔超平面最近的那些点
"""

from __future__ import print_function

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

import numpy


def load_dataset(filename):
    dataset = []
    labels = []
    with open(filename) as infile:
        for line in infile:
            datas = line.strip().split('\t')
            dataset.append([float(datas[0]), float(datas[1])])
            labels.append(float(datas[2]))
    return dataset, labels


def random_select_j(i, m):
    """ 返回任一 [0, m) 之间且不等于 i 的数 """
    j = i
    while j == i:
        j = int(numpy.random.uniform(0, m))
    return j


def adjust_alpha(aj, upper_bound, lower_bound):
    if aj > upper_bound:
        aj = upper_bound
    if lower_bound > aj:
        aj = lower_bound
    return aj


def estimate(alphas, labels, dataset, index, b):
    fx = float(
        numpy.multiply(alphas, labels).T
        * (dataset*dataset[index, :].T)
    ) + b
    e = fx - float(labels[index])
    return e


def smo_simple(dataset, labels, constant, toler, max_iter):
    """
    Platt的SMO算法简化版.
    = = = =
    每次循环中选择两个alpha进行优化处理.一旦找到一堆合适的alpha,
    那么就增大其中一个同时减少另外一个.
    * 两个alpha必须在间隔边界之外
    * 两个alpha还没有进行过区间化处理或者不在边界上

    Parameters
    ----------
    dataset
        数据集
    labels
        类型标签
    constant
        常数, 用于控制"最大化间隔"和"保证大部分点的函数间隔小于1.0"
    toler
        容错率
    max_iter
        最大循环次数

    Returns
    -------

    """
    dataset = numpy.mat(dataset)
    labels = numpy.mat(labels).T
    b = 0
    m, n = dataset.shape
    # 初始化alpha向量
    alphas = numpy.mat(numpy.zeros((m, 1)))
    num_iter = 0
    while num_iter < max_iter:
        # 对数据集中每个数据向量
        num_alpha_pairs_changed = False  # alpha 是否已经优化
        for i in range(m):
            # 计算 alpha[i] 的预测值, 估算其是否可以被优化
            Ei = estimate(alphas, labels, dataset, i, b)
            # 测试正/负间隔距离, alpha值, 是否满足KKT条件
            if not ((labels[i]*Ei < -toler and alphas[i] < constant)
                    or (labels[i]*Ei > toler and alphas[i] > 0)):
                logging.info('alpha[{0}]不需要调整.'.format(i))
                continue

            # 选择第二个 alpha[j]
            j = random_select_j(i, m)
            # alpha[j] 的预测值
            Ej = estimate(alphas, labels, dataset, j, b)

            # 保存旧值以便与调整后比较
            alphaI_old = alphas[i].copy()
            alphaJ_old = alphas[j].copy()

            # 计算 lower_bound/upper_bound, 调整 alpha[j] 至 (0, C) 之间
            if labels[i] != labels[j]:
                lower_bound = max(0, alphas[j] - alphas[i])
                upper_bound = min(constant, constant + alphas[j] - alphas[i])
            else:
                lower_bound = max(0, alphas[j] + alphas[i] - constant)
                upper_bound = min(constant, alphas[j] + alphas[i])
            if lower_bound == upper_bound:
                logging.warning('lower_bound == upper_bound == {0}'.format(lower_bound))
                continue

            # 计算 alpha[j] 的最优修改量
            deta = (
                2.0 * dataset[i, :] * dataset[j, :].T
                - dataset[i, :] * dataset[i, :].T
                - dataset[j, :] * dataset[j, :].T
            )
            # 如果deta==0, 则需要退出for循环的当前迭代过程.
            # 简化版中不处理这种少量出现的特殊情况
            if deta >= 0:
                logging.warning('{0}(deta) >= 0'.format(deta))
                continue

            # 计算新的 alpha[j]
            alphas[j] -= labels[j] * (Ei - Ej) / deta
            alphas[j] = adjust_alpha(alphas[j], upper_bound, lower_bound)
            deta_j = abs(alphas[j] - alphaJ_old)
            # 若 alpha[j] 的改变量太少, 不采用
            if deta_j < 0.00001:
                logging.warning('j 变化量太少, 不采用. ({0})'.format(deta_j))
                continue

            # 对 alpha[i] 做 alpha[j] 同样大小, 方向相反的改变
            alphas[i] += labels[j] * labels[i] * (alphaJ_old - alphas[j])

            # 给两个 alpha 值设置常量 b
            b1 = (
                b - Ei
                - labels[i] * (alphas[i]-alphaI_old) * dataset[i, :] * dataset[i, :].T
                - labels[j] * (alphas[j]-alphaJ_old) * dataset[i, :] * dataset[j, :].T
            )
            b2 = (
                b - Ej
                - labels[i] * (alphas[i]-alphaI_old) * dataset[i, :] * dataset[j, :].T
                - labels[j] * (alphas[j]-alphaJ_old) * dataset[j, :] * dataset[j, :].T
            )
            if 0 < alphas[i] < constant:
                b = b1
            elif 0 < alphas[j] < constant:
                b = b2
            else:
                b = (b1 + b2)/2.0

            num_alpha_pairs_changed = True
            logging.debug('numIter: {:d} i:{:d}, pairs changed {}'.format(
                num_iter, i, num_alpha_pairs_changed
            ))
        if num_alpha_pairs_changed == 0:
            num_iter += 1
        else:
            num_iter = 0
        logging.info('iteration number: {0}'.format(num_iter))
    return b, alphas

if __name__ == '__main__':
    dataset, labels = load_dataset('testSet.txt')
    length = len(labels)
    b, alphas = smo_simple(dataset, labels, 0.6, 0.001, 40)
    logging.info('支持向量:')
    logging.info([(dataset[i], labels[i]) for i in range(length) if alphas[i] > 0])


def kernelTrans(X, A, kTup):
    '''calc the kernel or transform data to a higher dimensional space'''
    m, n = numpy.shape(X)
    K = numpy.mat(numpy.zeros((m, 1)))
    if   kTup[0] == 'lin':
        K = X * A.T  #linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        # divide in NumPy is element-wise not matrix like Matlab
        K = numpy.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataset, labels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataset
        self.labels = labels
        self.C = C
        self.tol = toler
        self.m, self.n = shape(dataset)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labels).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labels[k])
    return Ek

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = random_select_j(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labels[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labels[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labels[i] != oS.labels[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labels[j]*(Ei - Ej)/eta
        oS.alphas[j] = AdjustAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labels[j]*oS.labels[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labels[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labels[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labels[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labels[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; isAlphaPairsChanged = 0
    while (iter < maxIter) and ((isAlphaPairsChanged > 0) or (entireSet)):
        isAlphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                isAlphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,isAlphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                isAlphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,isAlphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (isAlphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labels = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labels[i],X[i,:].T)
    return w

def testRbf(k1=1.3):
    dataArr,labelArr = LoadDataset('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labels = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labels[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = LoadDataset('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labels = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labels = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labels[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labels = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))


'''#######********************************
Non-Kernel VErsions below
'''#######********************************

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labels = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag

def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labels).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labels[k])
    return Ek

def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = random_select_j(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labels[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labels[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labels[i] != oS.labels[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labels[j]*(Ei - Ej)/eta
        oS.alphas[j] = AdjustAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labels[j]*oS.labels[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labels[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labels[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labels[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labels[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; isAlphaPairsChanged = 0
    while (iter < maxIter) and ((isAlphaPairsChanged > 0) or (entireSet)):
        isAlphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                isAlphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,isAlphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                isAlphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,isAlphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (isAlphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas
