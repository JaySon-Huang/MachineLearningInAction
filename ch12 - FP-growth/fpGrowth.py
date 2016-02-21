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
    def __init__(self, dataset, min_support_degree=1):
        self.min_support_degree = min_support_degree
        self.table = {}
        # 对每个元素出现次数进行计数
        for transaction in dataset:
            for item in transaction:
                self.table[item] = (
                    self.table.get(item, 0) + dataset[transaction]
                )
        # 删除出现次数少于 min_support_degree 的项
        self.table = {
            key: value for (key, value) in self.table.items()
            if value >= self.min_support_degree
        }
        frequent_items = set(self.table.keys())

        # 如果所有项都不频繁, 跳过下面的处理步骤
        if len(frequent_items) == 0:
            self.root = None
            self.table = None
            return

        # 扩展 headerTable 以便保存计数值以及指向每种类型第一个元素项的指针
        self.table = {
            key: TableItem(value, None) for (key, value) in self.table.items()
        }

        self.root = Node('Null Set', 1, None)
        for transaction, count in dataset.items():
            local_dataset = {}
            for item in transaction:  # put transaction items in order
                if item in frequent_items:
                    local_dataset[item] = self.table[item].count
            if len(local_dataset) > 0:
                ordered_items = [v[0] for v in sorted(
                    local_dataset.items(), key=lambda p: p[1],
                    reverse=True
                )]
                # populate tree with ordered freq itemset
                self.__update(ordered_items, self.root, count)

    @property
    def is_empty(self):
        return self.root is None

    def __update(self, items, root, count):
        if items[0] in root.children:
            # 如果已经在孩子列表中, 增加出现次数
            root.children[items[0]].inc(count)
        else:
            # 把结点添加到当前结点的子节点上
            root.children[items[0]] = Node(items[0], count, root)
            # 更新 table
            if self.table[items[0]].head is None:
                self.table[items[0]].head = root.children[items[0]]
            else:
                temp = self.table[items[0]].head
                while temp.nodeLink is not None:
                    temp = temp.nodeLink
                temp.nodeLink = root.children[items[0]]
        # call update() with remaining ordered items
        if len(items) > 1:
            self.__update(items[1:], root.children[items[0]], count)

    @staticmethod
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

    def mine(self, prefix=None):
        if prefix is None:
            prefix = set([])
        frequent_items = []
        # (sort header table)
        items = [
            pair[0] for pair in
            sorted(self.table.items(), key=lambda p: p[1])
        ]
        for item in items:
            new_frequent_set = prefix | {item}
            # print('finalFrequent Item: ', new_frequent_set)
            frequent_items.append(tuple(new_frequent_set))
            condition_pattern_bases = self.find_prefix_paths(
                item, self.table[item].head
            )
            # print('condition_pattern_bases :', item, condition_pattern_bases)
            # 2. construct cond FP-tree from cond. pattern base
            condition_tree = FrequentPatternTree(
                condition_pattern_bases,
                self.min_support_degree
            )
            # print('head from conditional tree: ', condition_table)
            if not condition_tree.is_empty:  # 3. mine cond. FP-tree
                # logging.debug('conditional tree for: {0}'.format(new_frequent_set))
                # condition_tree.display(1)
                sub_frequent_items = condition_tree.mine(new_frequent_set)
                frequent_items.extend(sub_frequent_items)
        return frequent_items


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


def main():
    import pprint
    dataset = load_fake_dataset()
    dataset = {frozenset(transaction): 1 for transaction in dataset}
    fp_tree = FrequentPatternTree(dataset, min_support_degree=3)
    logging.info(pprint.pformat(fp_tree.table))

    logging.info(FrequentPatternTree.find_prefix_paths('x', fp_tree.table['x'].head))
    logging.info(FrequentPatternTree.find_prefix_paths('z', fp_tree.table['z'].head))
    logging.info(FrequentPatternTree.find_prefix_paths('r', fp_tree.table['r'].head))

    frequent_items = fp_tree.mine()
    logging.info(pprint.pformat(frequent_items))

    dataset = []
    with open('kosarak.dat', 'r') as infile:
        for line in infile:
            dataset.append(line.split())
    dataset = {frozenset(transaction): 1 for transaction in dataset}
    logging.debug(len(dataset))
    min_support_degree = 100000
    fp_tree = FrequentPatternTree(dataset, min_support_degree)
    frequent_items = fp_tree.mine()
    logging.info(pprint.pformat(frequent_items))


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
