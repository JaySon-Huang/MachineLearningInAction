#!/usr/bin/env python
# encoding=utf-8

from __future__ import print_function

import logging

from numpy import *

TRACE = logging.DEBUG - 1
logging.basicConfig(
    level=logging.DEBUG,
    # level=TRACE,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)


def load_fake_dataset():
    return [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5],
    ]


def drop_unsupported_candidate_set(
        dataset, candidate_sets_k, min_support_degree):
    """
    '支持度': 数据集中包含该项集的记录所占的比例

    Parameters
    ----------
    dataset :
        数据集
    candidate_sets_k :
        候选项集集合
    min_support_degree :
        最小支持度

    Returns
    -------
    result, support_degrees :
        支持度 >= min_support_degree的频繁项集, 频繁项集的支持度
    """
    candidate_set_count = {}
    for transaction in dataset:
        for candidate_set in candidate_sets_k:
            if candidate_set.issubset(transaction):
                num = candidate_set_count.get(candidate_set, 0)
                candidate_set_count[candidate_set] = num + 1
    result = []
    support_degrees = {}
    for candidate_set in candidate_set_count:
        # 计算每一项的支持度
        support = 1.0 * candidate_set_count[candidate_set] / len(dataset)
        if support >= min_support_degree:
            result.insert(0, candidate_set)
        support_degrees[candidate_set] = support
    return result, support_degrees


def generate_candidate_sets_k(original_sets, k):
    if k == 1:
        """构建大小为1的所有候选项集合"""
        c1 = set([])
        for transaction in original_sets:
            for item in transaction:
                c1.add(item)
        return list(map(frozenset,
                    sorted(list(map(
                        lambda item: [item, ], c1)))))

    for one_set in original_sets:
        assert len(one_set) == k - 1
    candidate_sets_k = []
    for i in range(len(original_sets)):
        for j in range(i + 1, len(original_sets)):
            # 如果两个集合的前 k-2 个元素相同, 则将它们合并为一个大小为 k 的集合
            # 原因见书的 P208, 第二段
            if sorted(list(original_sets[i])[:k - 2]) \
                    == sorted(list(original_sets[j])[:k - 2]):
                candidate_sets_k.append(original_sets[i] | original_sets[j])
    return candidate_sets_k


def apriori(raw_dataset, min_support_degree=0.5):
    candidate_sets = generate_candidate_sets_k(raw_dataset, 1)
    dataset = list(map(set, raw_dataset))
    frequent_items, all_support_degree = drop_unsupported_candidate_set(
        dataset, candidate_sets, min_support_degree
    )
    all_frequent_items = [frequent_items, ]
    k = 2
    while len(all_frequent_items[k - 2]) > 0:
        candidate_sets = generate_candidate_sets_k(all_frequent_items[k - 2], k)
        frequent_items, support_degrees = drop_unsupported_candidate_set(
            dataset, candidate_sets, min_support_degree
        )
        all_support_degree.update(support_degrees)
        all_frequent_items.append(frequent_items)
        k += 1
    return all_frequent_items, all_support_degree


class Rule(object):
    def __init__(self, conditions, consequence, confidence_degree):
        self.conditions = list(conditions)
        self.consequence = list(consequence)
        self.confidence_degree = confidence_degree

    def __str__(self):
        return '{0} --> {1}, confidence: {2}'.format(
            self.conditions,
            self.consequence,
            self.confidence_degree
        )

    @classmethod
    def generate_rules(
            cls, frequent_sets, support_degrees,
            min_confidence_degree=0.7):
        rules = []
        # only get the sets with two or more items
        for i in range(1, len(frequent_sets)):
            for frequent_set in frequent_sets[i]:
                H1 = [frozenset([item]) for item in frequent_set]
                if i == 1:
                    legal_rules = cls.__rules_from_confidence_degree(
                        frequent_set, H1, support_degrees,
                        min_confidence_degree
                    )
                else:
                    legal_rules = cls.__rules_from_consequences(
                        frequent_set, H1, support_degrees,
                        min_confidence_degree
                    )
                rules.extend(legal_rules)
        return rules

    @classmethod
    def __rules_from_confidence_degree(
            cls, frequent_set, consequences, support_degrees,
            min_confidence_degree):
        """计算置信度
        '置信度': P -> H 的置信度为 support(P ∪ H) / support(P)
        'P ∪ H': P 与 H 的并集

        Parameters
        ----------
        frequent_set
        consequences
        support_degrees
        min_confidence_degree

        Returns
        -------

        """
        legal_rules = []
        for consequence in consequences:
            conditions = frequent_set - consequence
            # 计算置信度
            confidence = support_degrees[frequent_set] / support_degrees[conditions]
            if confidence >= min_confidence_degree:
                rule = Rule(
                    conditions, consequence,
                    confidence
                )
                legal_rules.append(rule)
                logging.debug(rule)
        return legal_rules

    @classmethod
    def __rules_from_consequences(
            cls, frequent_set, consequences, support_degrees,
            min_confidence_degree):
        # try further merging
        if len(frequent_set) <= len(consequences[0]) + 1:
            return None

        # create Hm+1 new candidates
        Hmp1 = generate_candidate_sets_k(consequences, len(consequences[0]) + 1)
        legal_rules = cls.__rules_from_confidence_degree(
            frequent_set, Hmp1, support_degrees,
            min_confidence_degree
        )
        legal_consequence = list(map(
            lambda rule: rule.consequence,
            legal_rules
        ))
        # need at least two sets to merge
        if len(legal_consequence) > 1:
            sub_rules = cls.__rules_from_consequences(
                frequent_set, legal_consequence, support_degrees,
                min_confidence_degree
            )
            if sub_rules is not None:
                legal_rules.extend(sub_rules)
        return legal_rules


def main():
    import pprint
    raw_dataset = load_fake_dataset()
    frequent_sets, support_degrees = apriori(
        raw_dataset, min_support_degree=0.5
    )
    logging.info('frequent_sets: {0}'.format(
        pprint.pformat(frequent_sets)
    ))
    logging.info('support_degrees: {0}'.format(
        pprint.pformat(support_degrees)
    ))

    rules = Rule.generate_rules(
        frequent_sets, support_degrees,
        min_confidence_degree=0.7
    )
    logging.info(rules)

if __name__ == '__main__':
    main()



# from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# #votesmart.apikey = 'get your api key first'
# def getActionIds():
#     actionIdList = []; billTitleList = []
#     fr = open('recent20bills.txt')
#     for line in fr.readlines():
#         billNum = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billNum) #api call
#             for action in billDetail.actions:
#                 if action.level == 'House' and \
#                 (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print('bill: %d has actionId: %d' % (billNum, actionId))
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             print("problem getting bill %d" % billNum)
#         sleep(1)                                      #delay to be polite
#     return actionIdList, billTitleList
#
# def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#     for billTitle in billTitleList:#fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}#list of items in each transaction (politician)
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print('getting votes for actionId: %d' % actionId)
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:
#                 if not transDict.has_key(vote.candidateName):
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except:
#             print("problem getting actionId: %d" % actionId)
#         voteCount += 2
#     return transDict, itemMeaning
