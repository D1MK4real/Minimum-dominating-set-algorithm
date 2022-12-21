import numpy as np
import pandas as pd
import scipy.stats as sps
from tqdm.notebook import tqdm
import warnings


def d(v, graph, IN, BN, LN, FL, FREE):
    if v in BN:
        # print(len(set([elem for elem in graph[v]]).intersection(FREE.union(FL))))
        return len(set([elem for elem in graph[v]]).intersection(FREE.union(FL)))
    if v in FREE:
        # print(len(set([elem for elem in graph[v]]).intersection(FREE.union(FL).union(BN))))
        return len(set([elem for elem in graph[v]]).intersection(FREE.union(FL).union(BN)))
    if v in FL:
        # print(len(set([elem for elem in graph[v]]).intersection(FREE.union(BN))))
        return len(set([elem for elem in graph[v]]).intersection(FREE.union(BN)))


def find_path(v, graph, IN, BN, LN, FL, FREE):
    nowv = v
    ans = []
    while True:
        nextv = N(nowv, graph).intersection(FREE).difference(set(ans))
        if len(nextv) == 1:
            nextv = nextv.pop()
            if nextv in FREE and d(nextv, graph, IN, BN, LN, FL, FREE) == 2:
                nowv = nextv
                ans.append(nextv)
            else:
                break
    return ans


def max_BN(graph, IN, BN, LN, FL, FREE):
    ans = 0
    v_ans = -1
    for v in BN:
        dv = d(v, graph, IN, BN, LN, FL, FREE)
        if dv > ans:
            v_ans = v
            ans = dv
    return v_ans


def dfs(v, free, freefl, graph, colors):
    for nextv in graph[v]:
        if nextv in freefl:
            if nextv in free and colors[nextv] == 0:
                colors[nextv] = 1
                dfs(nextv, free, freefl, graph, colors)
            else:
                colors[nextv] = 1


def unreach(free, fl, bn, graph, n):
    if len(bn) == 0:
        return False
    freefl = free.union(fl)
    if len(freefl) == 0:
        return True
    colors = [0 for _ in range(n)]
    for v in bn:
        dfs(v, free, freefl, graph, colors)
    for v, c in enumerate(colors):
        if v in freefl and c == 0:
            return False
    return True


def move_from_BN_to_IN(v, graph, IN, BN, LN, FL, FREE):
    BN_copy = BN.copy()
    BN_copy.remove(v)
    IN_copy = IN.copy()
    IN_copy.add(v)
    y = set([elem for elem in graph[v]]).intersection(FREE)
    y2 = set([elem for elem in graph[v]]).intersection(FL)
    FREE_copy = FREE.copy().difference(y)
    FL_copy = FL.copy().difference(y2)
    BN_copy = BN_copy.union(y)
    LN_copy = LN.copy().union(y2)
    return IN_copy, BN_copy, LN_copy, FL_copy, FREE_copy


def move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE):
    BN_copy = BN.copy()
    BN_copy.remove(v)
    LN_copy = LN.copy()
    LN_copy.add(v)
    return IN.copy(), BN_copy, LN_copy, FL.copy(), FREE.copy()


def move_from_FREE_to_LN(v, graph, IN, BN, LN, FL, FREE):
    FREE_copy = FREE.copy()
    FREE_copy.remove(v)
    LN_copy = LN.copy()
    LN_copy.add(v)
    return IN.copy(), BN.copy(), LN_copy, FL.copy(), FREE_copy


def move_from_FL_to_LN(v, graph, IN, BN, LN, FL, FREE):
    FL_copy = FL.copy()
    FL_copy.remove(v)
    LN_copy = LN.copy()
    LN_copy.add(v)

    return IN.copy(), BN.copy(), LN_copy, FL_copy, FREE.copy()


def move_set_from_FREE_to_FL(SET, graph, IN, BN, LN, FL, FREE):
    FREE_copy = FREE.copy().difference(SET)
    FL_copy = FL.copy().union(SET)
    return IN.copy(), BN.copy(), LN.copy(), FL_copy, FREE_copy


def move_set_from_BN_to_LN(SET, graph, IN, BN, LN, FL, FREE):
    BN_copy = BN.copy().difference(SET)
    LN_copy = LN.copy().union(SET)
    return IN.copy(), BN_copy, LN_copy, FL.copy(), FREE.copy()


def move_from_FREE_to_IN(v, graph, IN, BN, LN, FL, FREE):
    FREE_copy = FREE.copy()
    FREE_copy.remove(v)
    IN_copy = IN.copy()
    IN_copy.add(v)
    y = set([elem for elem in graph[v]]).intersection(FREE)
    y2 = set([elem for elem in graph[v]]).intersection(FL)
    FREE_copy = FREE_copy.difference(y)
    FL_copy = FL.copy().difference(y2)
    BN_copy = BN.copy().union(y)
    LN_copy = LN.copy().union(y2)
    return IN_copy, BN_copy, LN_copy, FL_copy, FREE_copy


def move_from_LN_to_IN(v, graph, IN, BN, LN, FL, FREE):
    LN_copy = LN.copy()
    LN_copy.remove(v)
    IN_copy = IN.copy()
    IN_copy.add(v)
    y = set([elem for elem in graph[v]]).intersection(FREE)
    y2 = set([elem for elem in graph[v]]).intersection(FL)
    FREE_copy = FREE.copy().difference(y)
    FL_copy = FL.copy().difference(y2)
    BN_copy = BN.copy().union(y)
    LN_copy = LN_copy.union(y2)
    return IN_copy, BN_copy, LN_copy, FL_copy, FREE_copy


def move_set_from_FREE_to_IN(SET, graph, IN, BN, LN, FL, FREE):
    FREE_copy = FREE.copy().difference(SET)
    IN_copy = IN.copy().union(SET)
    FL_copy = FL.copy()
    BN_copy = BN.copy()
    LN_copy = LN.copy()
    for v in SET:
        y = set([elem for elem in graph[v]]).intersection(FREE)
        y2 = set([elem for elem in graph[v]]).intersection(FL)
        FREE_copy = FREE_copy.difference(y)
        FL_copy = FL_copy.difference(y2)
        BN_copy = BN_copy.union(y)
        LN_copy = LN_copy.union(y2)

    return IN_copy, BN_copy, LN_copy, FL_copy, FREE_copy


def move_set_from_FREE_to_IN_special_BN(first, SET, graph, IN, BN, LN, FL, FREE):
    FREE_copy = FREE.copy().difference(SET)
    IN_copy = IN.copy().union(SET)
    FL_copy = FL.copy()
    BN_copy = BN.copy().difference({first})
    LN_copy = LN.copy()
    for v in (SET.union({first})):
        y = set([elem for elem in graph[v]]).intersection(FREE)
        y2 = set([elem for elem in graph[v]]).intersection(FL)
        FREE_copy = FREE_copy.difference(y)
        FL_copy = FL_copy.difference(y2)
        BN_copy = BN_copy.union(y)
        LN_copy = LN_copy.union(y2)

    return IN_copy, BN_copy, LN_copy, FL_copy, FREE_copy


def move_special_to_LN(v, graph, IN, BN, LN, FL, FREE):
    if v in LN:
        return IN.copy(), BN.copy(), LN.copy(), FL.copy(), FREE.copy()
    if v in FREE:
        return move_from_FREE_to_LN(v, graph, IN, BN, LN, FL, FREE)
    if v in BN:
        return move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE)


def move_special_to_IN(v, graph, IN, BN, LN, FL, FREE):
    if v in LN:
        return move_from_LN_to_IN(v, graph, IN, BN, LN, FL, FREE)
    if v in FREE:
        return move_from_FREE_to_IN(v, graph, IN, BN, LN, FL, FREE)
    if v in BN:
        return move_from_BN_to_IN(v, graph, IN, BN, LN, FL, FREE)


def N(v, graph):
    return set([elem for elem in graph[v]])


def N_SET(SET, graph):
    ans = set([])
    for v in SET:
        ans.union(set([elem for elem in graph[v]]))
    return ans.difference(SET)


def rec(graph, n, IN, BN, LN, FL, FREE):
    if not unreach(FREE, FL, BN, graph, n):
        return n, IN
    if set([i for i in range(n)]) == LN.union(IN):
        return len(IN), IN
    v = max_BN(graph, IN, BN, LN, FL, FREE)
    if v == -1:
        return len(IN), IN
    if d(v, graph, IN, BN, LN, FL, FREE) >= 3 or (
            d(v, graph, IN, BN, LN, FL, FREE) == 2 and len(set([elem for elem in graph[v]]).intersection(FL)) > 0):
        cort = move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE)
        ans1 = rec(graph, n, *cort)
        cort2 = move_from_BN_to_IN(v, graph, IN, BN, LN, FL, FREE)
        ans2 = rec(graph, n, *cort2)
        mas = [ans1[0], ans2[0]]
        graphs = [ans1[1], ans2[1]]
        k = np.argmin(mas)
        return mas[k], graphs[k]
    elif d(v, graph, IN, BN, LN, FL, FREE) == 2:
        x1, x2 = N(v, graph).intersection(FREE)
        dx1 = d(x1, graph, IN, BN, LN, FL, FREE)
        dx2 = d(x2, graph, IN, BN, LN, FL, FREE)
        if dx1 > dx2:
            x1, x2 = x2, x1
        if min(dx1, dx2) == 2:
            z, = set([elem for elem in graph[x1]]).difference(set([v]))
            if z in FREE:
                cort = move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE)
                ans1 = rec(graph, n, *cort)
                cort2 = move_from_BN_to_IN(v, graph, IN, BN, LN, FL, FREE)
                cort3 = move_from_BN_to_IN(x1, graph, *cort2)
                ans2 = rec(graph, n, *cort3)
                cort4 = move_from_BN_to_LN(x1, graph, *cort2)
                ans3 = rec(graph, n, *cort4)
                mas = [ans1[0], ans2[0], ans3[0]]
                graphs = [ans1[1], ans2[1], ans3[1]]
                k = np.argmin(mas)
                return mas[k], graphs[k]
            elif z in FL:
                cort = move_from_BN_to_IN(v, graph, IN, BN, LN, FL, FREE)
                ans1 = rec(graph, n, *cort)
                return ans1
        else:
            condition = True
            for el in set([elem for elem in graph[x1]]).intersection(set([elem for elem in graph[x2]])).intersection(
                    FL):
                if d(el, graph, IN, BN, LN, FL, FREE) < 3:
                    condition = False
            if condition and (N(x1, graph).intersection(N(x2, graph))).difference(FL) == {v}:
                cort = move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE)
                ans1 = rec(graph, n, *cort)
                cort2 = move_from_BN_to_IN(v, graph, IN, BN, LN, FL, FREE)
                cort3 = move_from_BN_to_IN(x1, graph, *cort2)
                ans2 = rec(graph, n, *cort3)
                cort4 = move_from_BN_to_LN(x1, graph, *cort2)
                cort5 = move_from_BN_to_IN(x2, graph, *cort4)
                ans3 = rec(graph, n, *cort5)
                cort6 = move_from_BN_to_LN(x2, graph, *cort4)
                cort7 = move_set_from_FREE_to_FL(N_SET({x1, x2}, graph).intersection(FREE), graph, *cort6)
                cort8 = move_set_from_BN_to_LN(N_SET({x1, x2}, graph).intersection(BN), graph, *cort7)
                ans4 = rec(graph, n, *cort8)
                mas = [ans1[0], ans2[0], ans3[0], ans4[0]]
                graphs = [ans1[1], ans2[1], ans3[1], ans4[1]]
                k = np.argmin(mas)
                return mas[k], graphs[k]
            else:
                cort = move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE)
                ans1 = rec(graph, n, *cort)
                cort2 = move_from_BN_to_IN(v, graph, IN, BN, LN, FL, FREE)
                cort3 = move_from_BN_to_IN(x1, graph, *cort2)
                ans2 = rec(graph, n, *cort3)
                cort4 = move_from_BN_to_LN(x1, graph, *cort2)
                cort5 = move_from_BN_to_IN(x2, graph, *cort4)
                ans3 = rec(graph, n, *cort5)
                mas = [ans1[0], ans2[0], ans3[0]]
                graphs = [ans1[1], ans2[1], ans3[1]]
                k = np.argmin(mas)
                return mas[k], graphs[k]
    elif d(v, graph, IN, BN, LN, FL, FREE) == 1:
        path = find_path(v, graph, IN, BN, LN, FL, FREE)
        z = (N(([v] + path)[-1], graph).difference(set(path).union({v})))
        if len(z) == 0:
            return n, IN
        zk = None
        while len(z) > 0:
            zk = z.pop()
            if zk in FL or zk in BN or zk in FREE:
                break

        z = zk
        if z in FL and d(z, graph, IN, BN, LN, FL, FREE) == 1:
            cort1 = move_set_from_FREE_to_IN_special_BN(v, set(path), graph, IN, BN, LN, FL, FREE)
            cort2 = move_special_to_LN(z, graph, *cort1)
            ans1 = rec(graph, n, *cort2)
            return ans1
        elif z in FL and d(z, graph, IN, BN, LN, FL, FREE) > 1:
            cort1 = move_set_from_FREE_to_IN_special_BN(v, set(path[:len(path) - 1]), graph, IN, BN, LN, FL, FREE)
            cort2 = move_special_to_LN(([v] + path)[-1], graph, *cort1)
            ans1 = rec(graph, n, *cort2)
            return ans1
        elif z in BN:
            cort2 = move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE)
            ans1 = rec(graph, n, *cort2)
            return ans1
        elif z in FREE:
            cort1 = move_set_from_FREE_to_IN_special_BN(v, set(path), graph, IN, BN, LN, FL, FREE)
            cort2 = move_special_to_IN(z, graph, *cort1)
            ans1 = rec(graph, n, *cort2)
            cort3 = move_from_BN_to_LN(v, graph, IN, BN, LN, FL, FREE)
            ans2 = rec(graph, n, *cort3)
            mas = [ans1[0], ans2[0]]
            graphs = [ans1[1], ans2[1]]
            k = np.argmin(mas)
            return mas[k], graphs[k]
    else:
        return len(IN), IN
    if not unreach(FREE, FL, BN, graph, n):
        return n, IN
    if set([i for i in range(n)]) == LN.union(IN):
        return len(IN), IN


n = int(input())
e = int(input())
matrix = [[0 for _ in range(n)] for _ in range(n)]
graph = [[] for _ in range(n)]
for _ in range(e):
    i, j = input().split(',')
    i = int(i)
    j = int(j)
    matrix[i][j] = 1
    matrix[j][i] = 1
    graph[i].append(j)
    graph[j].append(i)
ans = n
graphs = None
for v in tqdm(range(n)):
    IN = {v}
    BN = N(v, graph)
    LN = set([])
    FL = set([])
    FREE = set([_ for _ in range(n)]).difference(IN).difference(BN)
    ansi = rec(graph, n, IN, BN, LN, FL, FREE)
    if ansi[0] < ans:
        ans = ansi[0]
        graphs = ansi[1]
print(ans)
print(graphs)
