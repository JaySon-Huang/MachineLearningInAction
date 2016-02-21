#!/usr/bin/env python
# encoding=utf-8

from __future__ import print_function

import logging

TRACE = logging.DEBUG - 1
logging.basicConfig(
    level=logging.DEBUG,
    # level=TRACE,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)


class Node(object):
    def __init__(self, name, num_occur, parent):
        self.parent = parent
        self.name = name
        self.count = num_occur
        self.nodeLink = None
        self.children = {}
    
    def inc(self, num_occur):
        self.count += num_occur
        
    def display(self, depth=1):
        print('  ' * depth, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(depth + 1)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class TableItem(object):
    def __init__(self, count, head):
        self.count = count
        self.head = head

    def __str__(self):
        return '({0}, {1})'.format(self.count, self.head)

    def __repr__(self):
        return str(self)

    def __cmp__(self, other):
        if self.count != other.count:
            return cmp(self.count, other.count)
        else:
            return cmp(self.head, other.head)


class FrequentPatternTree(object):
    def __init__(self):
        pass


# create FP-tree from dataset but don't mine
def createTree(dataset, min_support_degree=1):
    headerTable = {}
    # 对每个元素出现次数进行计数
    for trans in dataset:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataset[trans]
    # 删除出现次数少于 min_support_degree 的项
    headerTable = {
        key: value for (key, value) in headerTable.items()
        if value >= min_support_degree
    }
    frequent_items = set(headerTable.keys())

    # 如果所有项都不频繁, 跳过下面的处理步骤
    if len(frequent_items) == 0:
        return None, None

    # 扩展 headerTable 以便保存计数值以及指向每种类型第一个元素项的指针
    headerTable = {
        key: TableItem(value, None) for (key, value) in headerTable.items()
    }

    root = Node('Null Set', 1, None)
    for transaction, count in dataset.items():
        local_dataset = {}
        for item in transaction:  # put transaction items in order
            if item in frequent_items:
                local_dataset[item] = headerTable[item].count
        if len(local_dataset) > 0:
            ordered_items = [v[0] for v in sorted(
                local_dataset.items(), key=lambda p: p[1],
                reverse=True
            )]
            # populate tree with ordered freq itemset
            updateTree(ordered_items, root, headerTable, count)
    return root, headerTable


def updateTree(items, tree, headerTable, count):
    if items[0] in tree.children:
        # 如果已经在孩子列表中, 增加出现次数
        tree.children[items[0]].inc(count)
    else:
        # 把结点添加到当前结点的子节点上
        tree.children[items[0]] = Node(items[0], count, tree)
        # 更新 headerTable
        if headerTable[items[0]].head is None:
            headerTable[items[0]].head = tree.children[items[0]]
        else:
            temp = headerTable[items[0]].head
            while temp.nodeLink is not None:
                temp = temp.nodeLink
            temp.nodeLink = tree.children[items[0]]
    # call updateTree() with remaining ordered items
    if len(items) > 1:
        updateTree(items[1:], tree.children[items[0]], headerTable, count)


def find_prefix_paths(element, node):
    paths = {}
    while node is not None:
        leaf = node
        prefix = []
        while leaf.parent is not None:
            prefix.append(leaf.name)
            leaf = leaf.parent
        if len(prefix) > 1:
            paths[frozenset(prefix[1:])] = node.count
        node = node.nodeLink
    return paths


def mineTree(
        tree, headerTable, min_support_degree,
        prefix, frequent_items):
    # (sort header table)
    items = [pair[0] for pair in sorted(headerTable.items(), key=lambda p: p[1])]
    for basePat in items:
        new_frequent_set = prefix | {basePat}
        # print('finalFrequent Item: ', new_frequent_set)
        frequent_items.append(new_frequent_set)
        condition_pattern_bases = find_prefix_paths(
            basePat, headerTable[basePat].head
        )
        # print('condition_pattern_bases :', basePat, condition_pattern_bases)
        # 2. construct cond FP-tree from cond. pattern base
        condition_tree, condition_table = createTree(
            condition_pattern_bases,
            min_support_degree=min_support_degree
        )
        # print('head from conditional tree: ', condition_table)
        if condition_table is not None:  # 3. mine cond. FP-tree
            # logging.debug('conditional tree for: {0}'.format(new_frequent_set))
            # condition_tree.display(1)
            mineTree(
                condition_tree, condition_table,
                min_support_degree, new_frequent_set, frequent_items
            )


def load_fake_dataset():
    dataset = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm'],
    ]
    return dataset


def init_dataset(dataset):
    result = {}
    for transaction in dataset:
        result[frozenset(transaction)] = 1
    return result


def main():
    # import pprint
    # dataset = load_fake_dataset()
    # dataset = init_dataset(dataset)
    # fp_tree, table = createTree(dataset, min_support_degree=3)
    # fp_tree.display()
    # logging.info(pprint.pformat(table))
    #
    # logging.info(find_prefix_paths('x', table['x'].head))
    # logging.info(find_prefix_paths('z', table['z'].head))
    # logging.info(find_prefix_paths('r', table['r'].head))
    #
    # frequent_items = []
    # mineTree(
    #     fp_tree, table,
    #     min_support_degree=3,
    #     prefix=set([]), frequent_items=frequent_items
    # )
    # # logging.info(pprint.pformat(frequent_items))
    # print(pprint.pformat(frequent_items))

    dataset = []
    with open('kosarak.dat', 'r') as infile:
        for line in infile:
            dataset.append(line.split())
    dataset = init_dataset(dataset)
    min_support_degree = 100000
    fp_tree, table = createTree(dataset, min_support_degree)
    frequent_items = []
    mineTree(
        fp_tree, table,
        min_support_degree, set([]), frequent_items
    )
    print(frequent_items)


if __name__ == '__main__':
    main()


# import twitter
# from time import sleep
# import re
#
# def textParse(bigString):
#     urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
#     listOfTokens = re.split(r'\W*', urlsRemoved)
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#
# def getLotsOfTweets(searchStr):
#     CONSUMER_KEY = ''
#     CONSUMER_SECRET = ''
#     ACCESS_TOKEN_KEY = ''
#     ACCESS_TOKEN_SECRET = ''
#     api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
#                       access_token_key=ACCESS_TOKEN_KEY,
#                       access_token_secret=ACCESS_TOKEN_SECRET)
#     #you can get 1500 results 15 pages * 100 per page
#     resultsPages = []
#     for i in range(1,15):
#         print("fetching page %d" % i)
#         searchResults = api.GetSearch(searchStr, per_page=100, page=i)
#         resultsPages.append(searchResults)
#         sleep(6)
#     return resultsPages
#
# def mineTweets(tweetArr, minSup=5):
#     parsedList = []
#     for i in range(14):
#         for j in range(100):
#             parsedList.append(textParse(tweetArr[i][j].text))
#     initSet = createInitSet(parsedList)
#     myFPtree, myHeaderTab = createTree(initSet, minSup)
#     myFreqList = []
#     mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
#     return myFreqList
#
# #minSup = 3
# #simpDat = loadSimpDat()
# #initSet = createInitSet(simpDat)
# #myFPtree, myHeaderTab = createTree(initSet, minSup)
# #myFPtree.disp()
# #myFreqList = []
# #mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
