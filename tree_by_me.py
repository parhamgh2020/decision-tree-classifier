import pandas as pd
import numpy as np
from math import log2


def entropy(s):
    n_whole = len(s)
    n_mbr_t = dict()  # number of members in each type
    for i in s:
        n_mbr_t[i] = n_mbr_t.get(i, 0) + 1
    entropy = 0
    for k, v in n_mbr_t.items():
        entropy += (-1) * (n_mbr_t[k] / n_whole) * log2(n_mbr_t[k] / n_whole)
    return entropy


def gain(df1, df2, label):
    gain_part1 = entrophy(df[label])
    n_df1, n_df2 = len(df1), len(df2)
    n_total = n_df1 + n_df2
    df1_entropy = entrophy(df1.label.values)
    df2_entropy = entrophy(df2.label.values)
    gain_part2 = (n_df1 / n_total * df1_entropy) + (n_df2 / n_total * df2_entropy)
    return gain_part1 - gain_part2


def find_agent(data, attr, label):
    thresholds = set(data[attr].values)
    if len(thresholds) > 1:
        thresholds.remove(max(thresholds))
    gains = dict()
    for i in thresholds:
        df1 = data[data[attr] <= i]
        df2 = data[data[attr] > i]
        gains[i] = gain(df1, df2, label)
    threshold, gain_value = -1, -1
    for k, v in gains.items():
        if v > gain_value:
            threshold, gain_value = k, v
    return threshold, gain_value


def find_best_col(data, attr, label):
    column, threshold, gain = -1, -1, -1
    for i in attr:
        t, g = find_agent(data, i, label)
        if g > gain:
            column, threshold, gain = i, t, g
    return column, threshold, gain


df = pd.read_csv('table_tree.csv')
column, threshold, gain = find_best_col(df, df.columns, 'label')
print(column, threshold, gain)


class Node:
    def __init__(self, data, attr, label):
        self.data = data
        self.column_name = None
        self.threshold = None
        self.gain = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = False
        self.leaf_label = None

        self.column_name, self.threshold, self.gain = \
            find_best_col(data, attr, label)


# class Tree:
#     def __init__(self):
#         self.root_node = None
#
#     def fit(self, x_train, y_train):
#         data = pd.concat([x_train, y_train], axis=1)
#         attr = x_train.columns
#         label = 'label'
#         # --
#         node = Node(data, attr, label)
#         if self.root_node is None:
#             self.root_node = node
#         else:
#             current = self.root_node
#             parent = None
#             while Tree
#                 parent = current
#                 filter_col = node.data[node.column_name]
#                     current = current.left_child
#                     if current is None:
#                         parent.left_child = node.data.iloc[i]
#                         continue
#                 else:
#                     current = current.right_child
#                     if current is None:
#                         parent.right_child = node.data.iloc[i]
#                         continue
#
#
# def predict(self, data):
#     ...
#
# # df = pd.read_csv('table_tree.csv')
# # tree = Tree()
# # tree.fit(df[['size', 'price', 'height']], df.label)
