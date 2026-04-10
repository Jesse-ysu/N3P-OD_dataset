# -*- coding: utf-8 -*-

from __future__ import division
from numpy import mean, std, array, argpartition, count_nonzero, empty, argsort
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from itertools import starmap, islice, cycle
from scipy.io import arff
import os, datetime
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.io import loadmat
from sklearn.metrics import auc, precision_score, recall_score, f1_score


def loadData(fileName, data_type, str):
    point_set = []
    for line in open(fileName, 'r'):
        point = [data_type(data) for data in line.split(str)]
        point_set.append(point)
    return array(point_set)


def dist(point1, point2):
    sum_dis = 0.0
    dimension = len(point1)
    for index in range(dimension):
        sum_dis += (point2[index] - point1[index]) ** 2 # 欧氏距离
    return sqrt(sum_dis)
    #     sum_dis = max(sum_dis, abs(point2[index] - point1[index])) # 切比雪夫距离
    # return sum_dis


# case2: -2: id  -1: outlier
def load_arff2(fileName):
    with open(fileName, 'r') as fh:
        dataset = arff.loadarff(fh)
        df = pd.DataFrame(dataset[0]) # DataFrame数组（保留列名）
        df['outlier'] = df['outlier'].str.decode('utf-8') # 最后一列是 bytes 类型，解码为字符串
        point_set = df.iloc[:, :-2].values.astype(float) # 剔除最后两列，并转为Numpy数组（丢失列名）
        labels = df.iloc[:, -1].values # 获取最后一列，并转为Numpy数组
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no':
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(int), outlier_num


# 加载 CSV 文件的函数
def load_csv(fileName):
    data = pd.read_csv(fileName)
    point_set = data.iloc[:, :-2].values.astype(float)
    labels = data.iloc[:, -1].values
    outlier_num = 0
    # for i, l in enumerate(labels):
    #     if l == 'no':
    #         labels[i] = 0
    #     else:
    #         labels[i] = 1
    #         outlier_num += 1
    return point_set, labels.astype(int), outlier_num


def scaled_mst(point_set):
    result_set = []
    adjencent = [[] for _ in range(len(point_set))] # 邻接矩阵
    nodes_finished = []
    nodes_unfinished = []
    ps_size = len(point_set)
    ratio_arr = [1] * ps_size
    dist_arr = [0] * ps_size
    edge_arr = [-1] * ps_size
    temp_dist1 = 1.0e14
    position = -1
    s = 0
    nodes_finished.append(s)
    for index in range(len(point_set)):
        if index == s:
            continue
        t = dist(point_set[s], point_set[index])
        if t == 0:
            result_set.append([s, index, 0])
            nodes_finished.append(index)
        else:
            dist_arr[index] = t
            edge_arr[index] = s
            ratio_arr[index] = t
            if t < temp_dist1:
                temp_dist1 = t
                position = index
    nodes_finished.append(position)
    result_set.append([s, position, temp_dist1])
    for index in range(len(point_set)):
        ratio_arr[index] = dist_arr[index] / temp_dist1
        if index not in nodes_finished:
            nodes_unfinished.append(index)
    q_index = 0
    while len(nodes_finished) < ps_size:
        min_ratio = 1.0e14
        for point_i in nodes_unfinished:
            new_node = nodes_finished[-1]
            d = dist(point_set[new_node], point_set[point_i])
            if d == 0:
                result_set.append([new_node, point_i, 0])
                nodes_finished.append(point_i)
                nodes_unfinished.remove(point_i)
                continue
            r = d / temp_dist1
            if r < ratio_arr[point_i]:
                dist_arr[point_i] = d
                ratio_arr[point_i] = r
                edge_arr[point_i] = new_node
            if ratio_arr[point_i] < min_ratio:
                min_ratio = ratio_arr[point_i]
                q_index = point_i
        temp_dist1 = dist_arr[q_index]
        nodes_finished.append(q_index)
        nodes_unfinished.remove(q_index)
        result_set.append([edge_arr[q_index], q_index, ratio_arr[q_index]])
    # # 计算邻接矩阵
    # for i in range(len(point_set)):
    #     for j in range(len(point_set)):
    #         if i == j:
    #             t = 1.0e14 # 矩阵对角线设为无限大
    #         else:
    #             t = dist(point_set[i], point_set[j]) # 计算距离
    #         adjencent[i].append([i, j, t])
    # # 更新距离
    # for i in range(len(result_set)):
    #     head = result_set[i][0] # 边的头节点
    #     tail = result_set[i][1] # 边的尾节点
    #     head_adjencent = min(adjencent[head], key=lambda x: x[2]) # 找到与头节点最近的点
    #     tail_adjencent = min(adjencent[tail], key=lambda x: x[2]) # 找到与尾节点最近的点
    #     d = dist(point_set[head], point_set[tail])
    #     if (head_adjencent[2] + tail_adjencent[2]) == 0:
    #         continue
    #     new_ratio_arr = 2 * d / (head_adjencent[2] + tail_adjencent[2]) # 最近邻缩放距离算法
    #     result_set[i][2] = new_ratio_arr # 更新距离
    return result_set, edge_arr, ratio_arr


def dfs(T, x, adjencent):
    for p in adjencent[x]:
        if p not in T:
            T.append(p)
            dfs(T, p, adjencent)


def cut_edge(edge, adjencent):
    Tu = []
    Tv = []
    adjencent[edge[0]].remove(edge[1])
    adjencent[edge[1]].remove(edge[0])
    Tu.append(edge[0])
    Tv.append(edge[1])
    dfs(Tu, edge[0], adjencent)
    dfs(Tv, edge[1], adjencent)
    return Tu, Tv


def get_mean_std(edge_set):
    sum_dist = 0
    std = 0
    n = len(edge_set)
    for edge in edge_set:
        sum_dist += edge[2]
    mean = sum_dist / n
    for edge in edge_set:
        std += abs(edge[2] - mean) ** 2
    std = sqrt(std)
    return mean + std


def cut_tree(tree, adjencent, clusters, largest_point):
    tu, tv = cut_edge(tree[0], adjencent)
    tree.remove(tree[0])
    left_tree = []
    right_tree = []
    if len(tu) > largest_point:
        for edge in tree:
            if edge[0] in tu or edge[1] in tu:
                left_tree.append(edge)
        cut_tree(left_tree, adjencent, clusters, largest_point)
    else:
        clusters.append(tu)
    if len(tv) > largest_point:
        for edge in tree:
            if edge[0] in tv or edge[1] in tv:
                right_tree.append(edge)
        cut_tree(right_tree, adjencent, clusters, largest_point)
    else:
        clusters.append(tv)


def CMOD(point_set):
    dimension = len(point_set[0])
    result_set, edge_arr, dist_arr = scaled_mst(point_set)
    sorted_edge = sorted(result_set, key=lambda x: x[2], reverse=True)
    data_size = len(point_set)
    least_point = sqrt(data_size / dimension)
    largest_point = data_size - least_point
    # ratio_threshold = get_mean_std(sorted_edge)
    # labels = [0] * data_size
    adjencent = [[] for i in range(data_size)]
    for edge in sorted_edge:
        adjencent[edge[0]].append(edge[1])
        adjencent[edge[1]].append(edge[0])
    clusters = []
    cut_tree(sorted_edge, adjencent, clusters, largest_point)
    cls_num = len(clusters)
    centroids = [0] * cls_num
    scores = [0] * len(point_set)
    for i, cl in enumerate(clusters):
        # 小于最小簇大小的簇，离群值设为无限大
        if len(cl) < least_point:
            for p in cl:
                scores[p] = 1e14
            continue
        temp_centroid = get_centroid(cl, point_set) # 找到中位数点
        temp_centroid = get_second_centroid(cl, point_set, temp_centroid) # 找到二次中位数点
        centroid_dist = 0 # 簇内距离
        for j, p in enumerate(cl):
            centroid_dist += dist(point_set[p], temp_centroid) # 计算簇内距离
        for p in cl:
            scores[p] = dist(point_set[p], temp_centroid) # 离群值设为点到中位数点的距离
    return scores


def scores2outliers(scores, outlier_num):
    # scores_arr = array(scores)
    # outliers = argpartition(scores_arr, outlier_num)
    # print(outliers[-outlier_num:])
    sorted_scores = sorted(scores, reverse=True)
    # print(sorted_scores)
    outliers = []
    # outliers = scores_arr.argmin(numberofvalues=outlier_num)
    for i in range(outlier_num):
        idx = scores.index(sorted_scores[i])
        scores[idx] = 0
        outliers.append(idx)
        # scores.remove(scores[idx])
    return outliers


def get_centroid(clusters, point_set):
    sum_dist = [0] * len(clusters)
    for i, p in enumerate(clusters):
        for j, q in enumerate(clusters):
            sum_dist[i] += dist(point_set[p], point_set[q]) # 计算簇内距离
    return point_set[clusters[sum_dist.index(min(sum_dist))]] # 找到距离最小的点作为中位数点


def get_second_centroid(clusters, point_set, first_centroid):
    drop_num = int(len(clusters) / 4) # 丢弃点数
    second_clusters = [0] * (len(clusters) - drop_num) # 二次簇
    cen_dist = [0] * len(clusters) # 二次簇距离
    for i, p in enumerate(clusters):
        cen_dist[i] = dist(point_set[p], first_centroid) # 计算簇距离
    for i in range(len(clusters) - drop_num):
        n = cen_dist.index(min(cen_dist)) # 找到距离最小的点
        second_clusters[i] = clusters[n] # 加入二次簇
        cen_dist.pop(n) # 删除该点
    return get_centroid(second_clusters, point_set) # 找到二次簇的中位数点


# case3: -2: outlier -1: id
def load_arff3(fileName):
    with open(fileName) as fh:
        dataset = array(arff.load(fh)['data'])
        point_set = dataset[:, :-2].astype(float)
        labels = dataset[:, -2]
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no':
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(int), outlier_num


# case4: 0 : id; -1: outlier
def load_arff4(fileName):
    with open(fileName) as fh:
        dataset = array(arff.load(fh)['data'])
        point_set = dataset[:, 1:-1].astype(float)
        labels = dataset[:, -1]
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no':
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(int), outlier_num


def get_centroid1_ps(clusters, point_set):
    centroids = [0] * len(point_set[0])
    for p in clusters:
        for i, ele in enumerate(point_set[p]):
            centroids[i] += ele
    center = [ce / len(clusters) for ce in centroids]
    return center


def get_centroid1(clusters):
    centroids = [0] * len(clusters[0])
    for p in clusters:
        for i, ele in enumerate(p):
            centroids[i] += ele
    center = [ce / len(clusters) for ce in centroids]
    return center


def get_centroid2(clusters):
    sum_dist = [0] * len(clusters)
    for i, p in enumerate(clusters):
        for j, q in enumerate(clusters):
            sum_dist[i] += dist(p, q)
    return clusters[sum_dist.index(min(sum_dist))].tolist()


def prim_mst(point_set):
    result_set = []
    nodes_finished = [];
    nodes_unfinished = []
    nodes_finished.append(0)
    dist_arr = [0] * len(point_set)
    edge_arr = [-1] * len(point_set)
    temp_dist1 = 1.0e14
    position = -1
    for index in range(len(point_set)):
        if index == 0:
            continue
        t = dist(point_set[0], point_set[index])
        dist_arr[index] = t
        edge_arr[index] = 0
        if t < temp_dist1:
            temp_dist1 = t
            position = index
    nodes_finished.append(position)
    result_set.append([0, position, temp_dist1])
    for index in range(len(point_set)):
        if index != 0 and index != position:
            nodes_unfinished.append(index)
    q_index = 0
    while len(nodes_unfinished) > 0:
        temp_dist2 = 1.0e14
        new_node = nodes_finished[-1]
        for point_i in nodes_unfinished:
            d = dist(point_set[new_node], point_set[point_i])
            if d < dist_arr[point_i]:  # and r != 0 :
                dist_arr[point_i] = d
                edge_arr[point_i] = new_node
            if dist_arr[point_i] < temp_dist2:
                temp_dist2 = dist_arr[point_i]
                q_index = point_i
        nodes_finished.append(q_index)
        nodes_unfinished.remove(q_index)
        result_set.append([edge_arr[q_index], q_index, dist_arr[q_index]])
    return result_set, edge_arr, dist_arr


def cent_score(center, point_set):
    scores = empty(len(point_set))
    for i, p in enumerate(point_set):
        print("p,center:", p, center)
        scores[i] = dist(p, center)
    print(center)
    print(scores)
    return argsort(scores)


if __name__ == "__main__":
    # fileName = "../data/data27.dat"      # 2
    # point_set = loadData(fileName, float, ',')

    # outlier_num = 5
    # scores = CMOD(point_set)
    # outliers = scores2outliers(scores, outlier_num)
    # print(outliers)

    p = r'./arff4' # 数据集

    f1 = open("result/method1_auc_%s.csv" % "WDBC", 'w') # 用于存储结果
    for root, dirs, files in os.walk(p):
        for name in files:
            fileName = os.path.join(p, name)
            file_name, file_type = os.path.splitext(name)
            if file_type == '.csv': # csv文件要特殊处理
                point_set, labels, outlier_num = load_csv(fileName)
            else:
                point_set, labels, outlier_num = load_arff2(fileName)
                # m = loadmat(fileName)
                # point_set = m["X"]
                # labels = m["y"].ravel()
            # point_set, labels, outlier_num = load_arff2(fileName)
            # point_set, labels, outlier_num = load_cls1o(fileName)
            # point_set = np.array(point_set.tolist()).astype(np.float)
            # labels = np.array(labels).astype(int)
            # print(file_name, len(point_set), outlier_num, len(point_set[0]))
            scores = CMOD(point_set) # 计算离群值
            fpr, tpr, thresholds = roc_curve(labels, scores) # 计算roc曲线
            roc_auc = auc(fpr, tpr) # 计算auc面积
            print(file_name, "%0.4f" % (roc_auc))
            f1.write(file_name + ',' + 'our' + ',' + str("%0.4f," % (roc_auc)) + '\n') # 写入结果文档