import heapq
from collections import deque


def plan_resources(records):
    n = len(records)
    if n == 0:
        return {"order": []}
    names = [r["name"] for r in records]
    order = sorted(range(n), key=lambda i: names[i])
    pos = {}
    for k, i in enumerate(order):
        pos[names[i]] = k
    recs = [records[i] for i in order]
    args = [None] * n
    ress = [None] * n
    for k in range(n):
        r = recs[k]
        a = r.get("argument_dependencies")
        args[k] = [pos[x] for x in a] if a else []
        b = r.get("result_dependencies")
        ress[k] = [pos[x] for x in b] if b else []
    argrev = [[] for _ in range(n)]
    for k in range(n):
        for d in args[k]:
            argrev[d].append(k)
    indet = [False] * n
    dq = deque()
    for k in range(n):
        if "indeterminate" in recs[k]:
            indet[k] = True
            dq.append(k)
    while dq:
        u = dq.popleft()
        for v in argrev[u]:
            if not indet[v]:
                indet[v] = True
                dq.append(v)
    N = 2 * n
    adj = [[] for _ in range(N)]
    for k in range(n):
        S = 2 * k
        F = 2 * k + 1
        adj[S].append(F)
        for d in args[k]:
            adj[2 * d + 1].append(S)
        if indet[k]:
            for d in ress[k]:
                adj[2 * d + 1].append(F)
        else:
            for d in ress[k]:
                adj[2 * d].append(F)
    for u in range(N):
        if len(adj[u]) > 1:
            adj[u] = sorted(set(adj[u]))
    indeg = [0] * N
    for u in range(N):
        for v in adj[u]:
            indeg[v] += 1
    heap = [u for u in range(N) if indeg[u] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    if len(out) == N:
        ev = []
        for u in out:
            ev.append([names[order[u >> 1]], "START" if u % 2 == 0 else "FINISH"])
        return {"order": ev}

    index = [-1] * N
    low = [0] * N
    onstk = [False] * N
    comp = [-1] * N
    stk = []
    counter = 0
    ncomp = 0
    for s in range(N):
        if index[s] != -1:
            continue
        work = [(s, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = counter
                low[v] = counter
                counter += 1
                stk.append(v)
                onstk[v] = True
            recurse = False
            av = adj[v]
            while pi < len(av):
                w = av[pi]
                pi += 1
                if index[w] == -1:
                    work[-1] = (v, pi)
                    work.append((w, 0))
                    recurse = True
                    break
                elif onstk[w]:
                    if index[w] < low[v]:
                        low[v] = index[w]
            if recurse:
                continue
            work[-1] = (v, pi)
            if low[v] == index[v]:
                while True:
                    w = stk.pop()
                    onstk[w] = False
                    comp[w] = ncomp
                    if w == v:
                        break
                ncomp += 1
            work.pop()
            if work:
                pv = work[-1][0]
                if low[v] < low[pv]:
                    low[pv] = low[v]

    csize = [0] * ncomp
    for u in range(N):
        csize[comp[u]] += 1
    m = -1
    for u in range(N):
        if csize[comp[u]] > 1:
            m = u
            break
        if u in adj[u]:
            m = u
            break
    c = comp[m]
    allowed = [False] * N
    nodes = [u for u in range(N) if comp[u] == c and u >= m]
    for u in nodes:
        allowed[u] = True
    radj = [[] for _ in range(N)]
    for u in nodes:
        for v in adj[u]:
            if allowed[v]:
                radj[v].append(u)
    INF = float("inf")
    dist = [INF] * N
    dist[m] = 0
    dq2 = deque([m])
    while dq2:
        u = dq2.popleft()
        for v in radj[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                dq2.append(v)
    best = INF
    for v in adj[m]:
        if allowed[v] and dist[v] + 1 < best:
            best = dist[v] + 1
    seq = [m]
    cur = m
    rem = best
    while rem > 1:
        nxt = -1
        for v in adj[cur]:
            if allowed[v] and dist[v] == rem - 1:
                nxt = v
                break
        seq.append(nxt)
        cur = nxt
        rem -= 1
    ev = []
    for u in seq:
        ev.append([names[order[u >> 1]], "START" if u % 2 == 0 else "FINISH"])
    return {"cycle": ev}