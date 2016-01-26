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

import numpy

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)


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
            if not ((labels[i] * Ei < -toler and alphas[i] < constant)
                    or (labels[i] * Ei > toler and alphas[i] > 0)):
                logging.debug('alpha[{0}]不需要调整.'.format(i))
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
                logging.debug('lower_bound == upper_bound == {0}'.format(lower_bound))
                continue

            # 计算 alpha[j] 的最优修改量
            delta = (
                2.0 * dataset[i, :] * dataset[j, :].T
                - dataset[i, :] * dataset[i, :].T
                - dataset[j, :] * dataset[j, :].T
            )
            # 如果 delta==0, 则需要退出for循环的当前迭代过程.
            # 简化版中不处理这种少量出现的特殊情况
            if delta >= 0:
                logging.warning('{0}(delta) >= 0'.format(delta))
                continue

            # 计算新的 alpha[j]
            alphas[j] -= labels[j] * (Ei - Ej) / delta
            alphas[j] = adjust_alpha(alphas[j], upper_bound, lower_bound)
            # 若 alpha[j] 的改变量太少, 不采用
            delta_j = abs(alphas[j] - alphaJ_old)
            if delta_j < 0.00001:
                logging.debug('j 变化量太少, 不采用. ({0})'.format(delta_j))
                continue

            # 对 alpha[i] 做 alpha[j] 同样大小, 方向相反的改变
            alphas[i] += labels[j] * labels[i] * (alphaJ_old - alphas[j])

            # 给两个 alpha 值设置常量 b
            b1 = (
                b - Ei
                - labels[i] * (alphas[i] - alphaI_old) * dataset[i, :] * dataset[i, :].T
                - labels[j] * (alphas[j] - alphaJ_old) * dataset[i, :] * dataset[j, :].T
            )
            b2 = (
                b - Ej
                - labels[i] * (alphas[i] - alphaI_old) * dataset[i, :] * dataset[j, :].T
                - labels[j] * (alphas[j] - alphaJ_old) * dataset[j, :] * dataset[j, :].T
            )
            if 0 < alphas[i] < constant:
                b = b1
            elif 0 < alphas[j] < constant:
                b = b2
            else:
                b = (b1 + b2) / 2.0

            num_alpha_pairs_changed = True
            logging.debug('numIter: {:d} i:{:d}, pairs changed {}'.format(
                num_iter, i, num_alpha_pairs_changed
            ))
        if num_alpha_pairs_changed == 0:
            num_iter += 1
        else:
            num_iter = 0
        logging.debug('iteration number: {0}'.format(num_iter))
    return b, alphas


def kernelTrans(X, A, kernel_info):
    """calc the kernel or transform data to a higher dimensional space
    `核函数` --

    Parameters
    ----------
    X
    A
    kernel_info : tuple
        包含核函数信息的元组

    Returns
    -------

    """
    m, n = numpy.shape(X)
    K = numpy.mat(numpy.zeros((m, 1)))
    if kernel_info[0] == 'lin':
        K = X * A.T  # linear kernel
    elif kernel_info[0] == 'rbf':  # radial bias function
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        # divide in NumPy is element-wise not matrix like Matlab
        K = numpy.exp(K / (-1 * kernel_info[1] ** 2))
    else:
        raise NameError('未定义的核函数')
    return K


class Options(object):
    def __init__(self, dataset, labels, constant, toler, kernel_info):
        self.X = dataset
        self.labels = labels
        self.constant = constant
        self.toler = toler
        self.m, self.n = dataset.shape
        self.alphas = numpy.mat(numpy.zeros((self.m, 1)))
        self.b = 0
        # eCache第一列表示该cache值是否有效
        self.eCache = numpy.mat(numpy.zeros((self.m, 2)))
        self.K = numpy.mat(numpy.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kernel_info)

    def updateEk(self, k):
        Ek = self.calc_estimate(k)
        self.eCache[k] = [1, Ek]

    def calc_estimate(self, index):
        from IPython import embed;embed()
        fx = float(
            numpy.multiply(self.alphas, self.labels).T * self.K[:, index]
            + self.b
        )
        e = fx - float(self.labels[index])
        return e

    def select_j(self, i, Ei):
        maxK = -1
        max_deltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]  # 设置第i个eCache缓存值
        validECaches = numpy.nonzero(self.eCache[:, 0].A)[0]
        if len(validECaches) > 1:
            # 在有效的缓存值中寻找deltaE最大的
            for k in validECaches:
                if k == i:
                    continue
                Ek = self.calc_estimate(k)
                deltaE = abs(Ei - Ek)
                if deltaE > max_deltaE:
                    maxK = k
                    max_deltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            # 没有任何有效的eCache缓存值 (如第一轮中)
            j = random_select_j(i, self.m)
            Ej = self.calc_estimate(j)
            return j, Ej


def inner_loop(i, options):
    # 计算 alpha[i] 的预测值, 估算其是否可以被优化
    Ei = options.calc_estimate(i)
    # 测试正/负间隔距离, alpha值, 是否满足KKT条件
    if not (((options.labels[i] * Ei < -options.toler) and (options.alphas[i] < options.constant))
            or ((options.labels[i] * Ei > options.toler) and (options.alphas[i] > 0))):
        logging.debug('alpha[{0}]不需要调整.'.format(i))
        return 0

    # 选择第二个 alpha[j], 并计算 alpha[j] 的预测值
    j, Ej = options.select_j(i, Ei)

    # 保存旧值以便与调整后比较
    alphaI_old = options.alphas[i].copy()
    alphaJ_old = options.alphas[j].copy()

    # 计算 lower_bound/upper_bound, 调整 alpha[j] 至 (0, C) 之间
    if options.labels[i] != options.labels[j]:
        lower_bound = max(0, options.alphas[j] - options.alphas[i])
        upper_bound = min(options.constant, options.constant + options.alphas[j] - options.alphas[i])
    else:
        lower_bound = max(0, options.alphas[j] + options.alphas[i] - options.constant)
        upper_bound = min(options.constant, options.alphas[j] + options.alphas[i])
    if lower_bound == upper_bound:
        logging.debug('lower_bound == upper_bound == {0}'.format(lower_bound))
        return 0

    # 计算 alpha[j] 的最优修改量
    delta = 2.0 * options.K[i, j] - options.K[i, i] - options.K[j, j]
    if delta >= 0:
        logging.warning('{0}(delta) >= 0'.format(delta))
        return 0

    # 计算新的 alpha[j]
    options.alphas[j] -= options.labels[j] * (Ei - Ej) / delta
    options.alphas[j] = adjust_alpha(options.alphas[j], upper_bound, lower_bound)
    options.updateEk(j)  # 更新缓存中Ej的值
    # 若 alpha[j] 的改变量太少, 不采用
    delta_j = abs(options.alphas[j] - alphaJ_old)
    if delta_j < 0.00001:
        logging.debug('j 变化量太少, 不采用. ({0})'.format(delta_j))
        return 0

    # 对 alpha[i] 做 alpha[j] 同样大小, 方向相反的改变
    options.alphas[i] += options.labels[j] * options.labels[i] * (alphaJ_old - options.alphas[j])
    options.updateEk(i)  # 更新缓存中Ei的值
    # 给两个 alpha 值设置常量 b
    b1 = (
        options.b - Ei
        - options.labels[i] * (options.alphas[i] - alphaI_old) * options.K[i, i]
        - options.labels[j] * (options.alphas[j] - alphaJ_old) * options.K[i, j]
    )
    b2 = (
        options.b - Ej
        - options.labels[i] * (options.alphas[i] - alphaI_old) * options.K[i, j]
        - options.labels[j] * (options.alphas[j] - alphaJ_old) * options.K[j, j]
    )
    if 0 < options.alphas[i] < options.constant:
        options.b = b1
    elif 0 < options.alphas[j] < options.constant:
        options.b = b2
    else:
        options.b = (b1 + b2) / 2.0
    return 1


def smoP(dataset, labels, constant, toler, max_iter, kernel_info=('lin', 0)):
    options = Options(
        numpy.mat(dataset),
        numpy.mat(labels).T,
        constant, toler, kernel_info
    )
    num_iter = 0
    scan_entire_set = True
    num_alpha_pairs_changed = 0
    while (num_iter < max_iter) and ((num_alpha_pairs_changed > 0) or scan_entire_set):
        num_alpha_pairs_changed = 0

        if scan_entire_set:
            # 遍历alpha, 使用 `inner_loop` 选择 alpha-j, 并在可能是对其进行优化
            for i in range(options.m):
                num_alpha_pairs_changed += inner_loop(i, options)
                logging.debug('scanning : num_iter({}) i({}) pairs changed({})'.format(
                    num_iter, i, num_alpha_pairs_changed
                ))
            num_iter += 1
        else:
            # 遍历所有非边界(不在边界0或C上)的 alpha
            non_bound_indexs = numpy.nonzero(
                (options.alphas.A > 0) * (options.alphas.A < constant)
            )[0]
            for i in non_bound_indexs:
                num_alpha_pairs_changed += inner_loop(i, options)
                logging.debug('non-bound : num_iter({}) i({}) pairs changed({})'.format(
                    num_iter, i, num_alpha_pairs_changed
                ))
            num_iter += 1

        if scan_entire_set:
            scan_entire_set = False
        elif num_alpha_pairs_changed == 0:
            scan_entire_set = True
        logging.debug('iteration number: {}'.format(num_iter))
    return options.b, options.alphas


def get_weights(alphas, dataset, labels):
    dataset = numpy.mat(dataset)
    labels = numpy.mat(labels).T
    m, n = dataset.shape
    w = numpy.zeros((n, 1))
    for i in range(m):
        w += numpy.multiply(alphas[i] * labels[i], dataset[i, :].T)
    return w


def test_rbf(k1=1.3):
    import pprint
    dataset, labels = load_dataset('testSetRBF.txt')
    b, alphas = smoP(dataset, labels, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    dataset = numpy.mat(dataset)
    labels = numpy.mat(labels).T
    support_vectors_index = tuple(numpy.nonzero(alphas.A > 0))[0]
    support_vectors = dataset[support_vectors_index]
    support_vectors_label = labels[support_vectors_index]
    m, _n = support_vectors.shape
    logging.info('支持向量 ({})个:'.format(m))
    logging.info(pprint.pformat(zip(
        support_vectors.tolist(), support_vectors_label.A1.tolist()
    )))

    m, _n = dataset.shape
    errorCount = 0
    for i in range(m):
        # 利用 核函数 && 支持向量 进行分类.
        kernelEval = kernelTrans(support_vectors, dataset[i, :], ('rbf', k1))
        predict = (
            kernelEval.T
            * numpy.multiply(support_vectors_label, alphas[support_vectors_index])
            + b
        )
        if numpy.sign(predict) != numpy.sign(labels[i]):
            errorCount += 1
    logging.info('训练集上错误率: {:.2%}'.format(1.0 * errorCount / m))

    # 使用训练出来的SVM来对测试集进行分类, 检查错误率
    dataset, labels = load_dataset('testSetRBF2.txt')
    dataset = numpy.mat(dataset)
    labels = numpy.mat(labels).T
    m, _n = dataset.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(support_vectors, dataset[i, :], ('rbf', k1))
        predict = (
            kernelEval.T
            * numpy.multiply(support_vectors_label, alphas[support_vectors_index])
            + b
        )
        if numpy.sign(predict) != numpy.sign(labels[i]):
            errorCount += 1
    logging.info('测试集上错误率: {:.2%}'.format(1.0 * errorCount / m))

""" 使用SVM来进行手写数字识别 """


def img2vector(filename):
    vector = numpy.zeros((1, 1024))
    with open(filename) as infile:
        for lineno, line in enumerate(infile):
            for rowno in range(32):
                vector[0, 32 * lineno + rowno] = int(line[rowno])
        return vector


def load_images(dir_name):
    import os
    files = os.listdir(dir_name)
    labels = []
    dataset = numpy.zeros((len(files), 1024))
    for i, filename in enumerate(files):
        name = os.path.splitext(filename)[0]
        class_num = int(name.split('_')[0])
        if class_num == 9:
            labels.append(-1)
        elif class_num == 1:
            labels.append(1)
        else:
            raise ValueError('本分类器为二分类器, 不支持除1/9外的数字')
        dataset[i, :] = img2vector('%s/%s' % (dir_name, filename))
    return dataset, labels


def test_digits(kernel_info=('rbf', 10)):
    dataset, labels = load_images('digits/trainingDigits')
    dataset = numpy.mat(dataset)
    labels = numpy.mat(labels).T

    b, alphas = smoP(dataset, labels, 200, 0.0001, 10000, kernel_info)
    support_vectors_index = numpy.nonzero(alphas.A > 0)[0]
    support_vectors = dataset[support_vectors_index]
    support_vectors_label = labels[support_vectors_index]
    m, _n = support_vectors.shape
    import pprint
    logging.info('支持向量 ({})个:'.format(m))
    logging.info(pprint.pformat(zip(
        support_vectors.tolist(), support_vectors_label.A1.tolist()
    )))

    m, n = dataset.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(support_vectors, dataset[i, :], kernel_info)
        predict = (
            kernelEval.T
            * numpy.multiply(support_vectors_label, alphas[support_vectors_index])
            + b
        )
        if numpy.sign(predict) != numpy.sign(labels[i]):
            errorCount += 1
    logging.info('训练集上错误率: {:.2%}'.format(1.0 * errorCount / m))

    dataset, labels = load_images('digits/testDigits')
    dataset = numpy.mat(dataset)
    labels = numpy.mat(labels).T
    errorCount = 0
    m, n = dataset.shape
    for i in range(m):
        kernelEval = kernelTrans(support_vectors, dataset[i, :], kernel_info)
        predict = kernelEval.T * numpy.multiply(support_vectors_label, alphas[support_vectors_index]) + b
        if numpy.sign(predict) != numpy.sign(labels[i]):
            errorCount += 1
    logging.info('测试集上错误率: {:.2%}'.format(1.0 * errorCount / m))

""" main 函数 """


def main():
    # import pprint
    # dataset, labels = load_dataset('testSet.txt')
    # length = len(labels)
    # b, alphas = smo_simple(dataset, labels, 0.6, 0.001, 40)
    # logging.info('支持向量:')
    # logging.info(pprint.pformat(
    #     [(dataset[i], labels[i]) for i in range(length) if alphas[i] > 0]
    # ))

    # 使用核函数的SVM
    test_rbf(k1=1.3)

    # 手写数字识别
    test_digits()

if __name__ == '__main__':
    main()


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
            Ek = calc_estimate(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = random_select_j(i, oS.m)
        Ej = calc_estimate(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calc_estimate(oS, k)
    oS.eCache[k] = [1,Ek]

def innerLK(i, oS):
    Ei = calc_estimate(oS, i)
    if ((oS.labels[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labels[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = select_j(i, oS, Ei) #this has been changed from selectJrand
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
    oS = Options(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
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
