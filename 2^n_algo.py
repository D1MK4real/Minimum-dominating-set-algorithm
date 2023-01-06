import numpy as np
import pandas as pd
import scipy.stats as sps
import copy
from tqdm.notebook import tqdm
import warnings
import networkx as nx
import time


def dfs(v, graph, colors, vert):
    for nextv in graph[v]:
        if colors[nextv] == 0 and nextv in vert:
            colors[nextv] = 1
            dfs(nextv, graph, colors, vert)


def unreach(graph, vert, n):
    colors = [1 for _ in range(n)]
    for v in vert:
        colors[v] = 0
    first = vert[0]
    colors[first] = 1
    dfs(first, graph, colors, vert)
    for c in colors:
        if c == 0:
            return False
    return True


def check_conn(graph, vert, n):
    return unreach(graph, vert, n)


def check_dom(graph, vert, n):
    colors = [0 for _ in range(n)]
    for v in vert:
        colors[v] = 1
    for v in vert:
        for u in graph[v]:
            if colors[u] == 0 and u not in vert:
                colors[u] = 1
    for c in colors:
        if c == 0:
            return False
    return True


def from_matrix_to_list(matrix):
    n = len(matrix[0])
    graph1 = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if matrix[i][j] == 1:
                graph1[i].append(j)
                graph1[j].append(i)
    return graph1


n = int(input())
e = int(input())
matrix = [[0 for _ in range(n)] for _ in range(n)]
graph1 = [[] for _ in range(n)]
for _ in range(e):
    i, j = input().split(',')
    i = int(i)
    j = int(j)
    matrix[i][j] = 1
    matrix[j][i] = 1
    graph1[i].append(j)
    graph1[j].append(i)
#
# n = 22
# matrix = [[] for _ in range(n)]
# for i in range(n):
#     inp = input().split(',')
#     str = [int(x) for x in inp]
#     matrix[i] = str
#
# graph1 = from_matrix_to_list(matrix)
# print(len(matrix))

ans = n
graphs = None
found = False
start_time = time.time()
for i in tqdm(range(2 ** n)):
    out = [1 if i & (1 << (n - 1 - k)) else 0 for k in range(n)]
    vert = []
    for i in range(n):
        if out[i] == 1:
            vert.append(i)
    if check_dom(graph1, vert, n) and check_conn(graph1, vert, n):
        if len(vert) < ans:
            ans = len(vert)
            graphs = vert
print('Размер множества: ', ans)
print('Искомое множество вершин', graphs)
print('Время работы: ', time.time() - start_time, 'c.')
