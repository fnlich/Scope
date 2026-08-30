from itertools import combinations
from collections import deque


def configure_routes(width, availability, capacities, routes):
    P = len(availability)
    n = len(routes)
    if n == 0:
        return ((), tuple([0] * P), 0, 0)
    if P == 0:
        return None
    per = []
    for r in routes:
        d = {}
        for pos in range(len(r)):
            c = r[pos]
            g = c[0]
            cst = c[1]
            if g < 0:
                continue
            p = g // width
            b = g % width
            if p >= P:
                continue
            if not ((availability[p] >> b) & 1):
                continue
            key = (p, b)
            v = (cst, pos)
            if key not in d or v < d[key]:
                d[key] = v
        if not d:
            return None
        per.append(d)
    pinset = set()
    for d in per:
        for k in d:
            pinset.add(k)
    pins = sorted(pinset)
    pinidx = {}
    for j in range(len(pins)):
        pinidx[pins[j]] = j
    m = len(pins)
    B = 601
    M = B ** n
    W = [B ** (n - 1 - i) for i in range(n)]
    S = 0
    T = 1 + n + m + P
    V = T + 1
    graph = [[] for _ in range(V)]
    eto = []
    ecap = []
    ecost = []

    def ae(u, v, cap, cost):
        idx = len(eto)
        eto.append(v)
        ecap.append(cap)
        ecost.append(cost)
        graph[u].append(idx)
        eto.append(u)
        ecap.append(0)
        ecost.append(-cost)
        graph[v].append(idx + 1)
        return idx

    for i in range(n):
        ae(S, 1 + i, 1, 0)
    sig_edges = []
    for i in range(n):
        for key in sorted(per[i]):
            cst, pos = per[i][key]
            j = pinidx[key]
            idx = ae(1 + i, 1 + n + j, 1, cst * M + pos * W[i])
            sig_edges.append((idx, i, j, pos, cst))
    for j in range(m):
        p = pins[j][0]
        ae(1 + n + j, 1 + n + m + p, 1, 0)
    portsink = [-1] * P
    for p in range(P):
        portsink[p] = ae(1 + n + m + p, T, 0, 0)
    base = list(ecap)
    best = None
    answer = None
    for k in range(1, P + 1):
        found = False
        for sub in combinations(range(P), k):
            for e in range(len(ecap)):
                ecap[e] = base[e]
            for p in sub:
                ecap[portsink[p]] = capacities[p]
            flow = 0
            total = 0
            while flow < n:
                dist = [None] * V
                dist[S] = 0
                inq = [False] * V
                preve = [-1] * V
                q = deque()
                q.append(S)
                inq[S] = True
                while q:
                    u = q.popleft()
                    inq[u] = False
                    du = dist[u]
                    if du is None:
                        continue
                    for e in graph[u]:
                        if ecap[e] > 0:
                            v = eto[e]
                            nd = du + ecost[e]
                            if dist[v] is None or nd < dist[v]:
                                dist[v] = nd
                                preve[v] = e
                                if not inq[v]:
                                    inq[v] = True
                                    q.append(v)
                if dist[T] is None:
                    break
                f = n - flow
                v = T
                while v != S:
                    e = preve[v]
                    if ecap[e] < f:
                        f = ecap[e]
                    v = eto[e ^ 1]
                v = T
                while v != S:
                    e = preve[v]
                    ecap[e] -= f
                    ecap[e ^ 1] += f
                    v = eto[e ^ 1]
                flow += f
                total += f * dist[T]
            if flow != n:
                continue
            found = True
            if best is None or total < best:
                best = total
                sel = [None] * n
                for idx, i, j, pos, cst in sig_edges:
                    if ecap[idx] < base[idx]:
                        sel[i] = (pos, cst, j)
                answer = sel
        if found:
            break
    if answer is None:
        return None
    masks = [0] * P
    choices = []
    totalcost = 0
    for i in range(n):
        pos, cst, j = answer[i]
        p, b = pins[j]
        masks[p] |= (1 << b)
        choices.append(pos)
        totalcost += cst
    touched = 0
    for p in range(P):
        if masks[p]:
            touched += 1
    return (tuple(choices), tuple(masks), touched, totalcost)