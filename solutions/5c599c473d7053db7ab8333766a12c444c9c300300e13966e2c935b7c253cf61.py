from collections import deque


def plan_provenance(variants, rewrite_only):
    K2 = 10 ** 6
    K1 = 10 ** 12
    rw = set(rewrite_only)

    scope_cost = {}
    scope_tr = {}
    scope_conf = {}
    var_scopes = []
    resum = []

    for v in variants:
        root = v[0]
        repo = v[1]
        tr = v[2]
        vb = v[3]
        blobs = v[4]
        r = tr not in rw
        resum.append((r, vb))
        ks = []
        for b in blobs:
            h = b[0]
            sb = b[1]
            key = (root, repo, h)
            if key not in scope_cost:
                scope_cost[key] = sb
                scope_tr[key] = tr
                scope_conf[key] = False
            else:
                if scope_tr[key] != tr:
                    scope_conf[key] = True
            if r:
                ks.append(key)
        var_scopes.append(ks)

    eligible = {}
    for key in scope_cost:
        eligible[key] = (not scope_conf[key]) and (scope_tr[key] not in rw)

    base = 0
    nodes = []
    node_w = []
    for i in range(len(variants)):
        r, vb = resum[i]
        ks = var_scopes[i]
        if not r or not ks:
            continue
        w = vb * K1 + K2 + 1
        ok = True
        for k in ks:
            if not eligible[k]:
                ok = False
                break
        if ok:
            nodes.append(i)
            node_w.append(w)
        else:
            base += w

    if not nodes:
        total = base
        B = total // K1
        M = (total // K2) % K2
        V = total % K2
        return [B, M, V, M - V]

    scope_id = {}
    for i in nodes:
        for k in var_scopes[i]:
            if k not in scope_id:
                scope_id[k] = len(scope_id)

    nvar = len(nodes)
    nsc = len(scope_id)
    N = 2 + nvar + nsc
    S = 0
    T = 1

    to = []
    cap = []
    graph = [[] for _ in range(N)]

    def add(u, vtx, c):
        graph[u].append(len(to))
        to.append(vtx)
        cap.append(c)
        graph[vtx].append(len(to))
        to.append(u)
        cap.append(0)

    INF = 1 << 90
    total_w = 0
    for idx in range(nvar):
        w = node_w[idx]
        total_w += w
        add(S, 2 + idx, w)
        for k in var_scopes[nodes[idx]]:
            add(2 + idx, 2 + nvar + scope_id[k], INF)
    for k, sid in scope_id.items():
        add(2 + nvar + sid, T, scope_cost[k] * K1 + K2)

    level = [0] * N
    it = [0] * N

    def bfs():
        for i in range(N):
            level[i] = -1
        level[S] = 0
        q = deque()
        q.append(S)
        while q:
            u = q.popleft()
            for eid in graph[u]:
                if cap[eid] > 0:
                    vv = to[eid]
                    if level[vv] < 0:
                        level[vv] = level[u] + 1
                        q.append(vv)
        return level[T] >= 0

    flow = 0
    while bfs():
        for i in range(N):
            it[i] = 0
        while True:
            stack = [S]
            path = []
            found = False
            while stack:
                u = stack[-1]
                if u == T:
                    found = True
                    break
                advanced = False
                gu = graph[u]
                while it[u] < len(gu):
                    eid = gu[it[u]]
                    vv = to[eid]
                    if cap[eid] > 0 and level[vv] == level[u] + 1:
                        stack.append(vv)
                        path.append(eid)
                        advanced = True
                        break
                    it[u] += 1
                if not advanced:
                    level[u] = -1
                    stack.pop()
                    if path:
                        path.pop()
                    if stack:
                        it[stack[-1]] += 1
            if not found:
                break
            f = cap[path[0]]
            for e in path:
                if cap[e] < f:
                    f = cap[e]
            for e in path:
                cap[e] -= f
                cap[e ^ 1] += f
            flow += f

    total = base + total_w - flow
    B = total // K1
    M = (total // K2) % K2
    V = total % K2
    return [B, M, V, M - V]