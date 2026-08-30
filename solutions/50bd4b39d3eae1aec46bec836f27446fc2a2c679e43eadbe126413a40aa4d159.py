import heapq


def render_plan(items, active_filter):
    idx = {}
    for rec in items:
        h = rec[0]
        if h in idx:
            return ("error", "duplicate", h)
        idx[h] = len(idx)
    n = len(items)
    for rec in items:
        h = rec[0]
        par = rec[1]
        src = rec[5]
        if par is not None and par not in idx:
            return ("error", "missing", h, "parent", par)
        if src is not None and src not in idx:
            return ("error", "missing", h, "source", src)
        for t in rec[2]:
            if t not in idx:
                return ("error", "missing", h, "before", t)

    def find_bad(nxt):
        st = [0] * n
        for i in range(n):
            if st[i]:
                continue
            path = []
            u = i
            while True:
                if u is None:
                    res = 2
                    break
                s = st[u]
                if s == 1 or s == 3:
                    res = 3
                    break
                if s == 2:
                    res = 2
                    break
                st[u] = 1
                path.append(u)
                u = nxt[u]
            for v in path:
                st[v] = res
        return [i for i in range(n) if st[i] == 3]

    parent = [idx[r[1]] if r[1] is not None else None for r in items]
    bad = find_bad(parent)
    if bad:
        return ("error", "parent_cycle", tuple(items[i][0] for i in bad))

    children = [[] for _ in range(n + 1)]
    for i in range(n):
        p = parent[i]
        children[p if p is not None else n].append(i)

    tin = [0] * (n + 1)
    tout = [0] * (n + 1)
    st = [n]
    ptr = [0]
    tin[n] = 0
    timer = 1
    while st:
        u = st[-1]
        i = ptr[-1]
        cu = children[u]
        if i < len(cu):
            ptr[-1] += 1
            v = cu[i]
            tin[v] = timer
            timer += 1
            st.append(v)
            ptr.append(0)
        else:
            tout[u] = timer
            timer += 1
            st.pop()
            ptr.pop()

    LOG = max(1, (n + 1).bit_length())
    up = [[0] * (n + 1) for _ in range(LOG)]
    u0 = up[0]
    for v in range(n):
        p = parent[v]
        u0[v] = p if p is not None else n
    u0[n] = n
    for k in range(1, LOG):
        uk = up[k]
        pk = up[k - 1]
        for v in range(n + 1):
            uk[v] = pk[pk[v]]

    def anc(a, b):
        return tin[a] <= tin[b] and tout[b] <= tout[a]

    def climb(a, b):
        u = a
        for k in range(LOG - 1, -1, -1):
            w = up[k][u]
            if not anc(w, b):
                u = w
        return u

    for i in range(n):
        h = items[i][0]
        for t in items[i][2]:
            j = idx[t]
            if i == j or anc(i, j) or anc(j, i):
                return ("error", "overlap", h, t)

    srcs = [idx[r[5]] if r[5] is not None else None for r in items]
    bad = find_bad(srcs)
    if bad:
        return ("error", "source_cycle", tuple(items[i][0] for i in bad))

    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for t in items[i][2]:
            j = idx[t]
            a = climb(i, j)
            b = climb(j, i)
            adj[a].append(b)
            indeg[b] += 1

    ordered = [None] * (n + 1)
    for p in range(n + 1):
        grp = children[p]
        if not grp:
            ordered[p] = []
            continue
        heap = [c for c in grp if indeg[c] == 0]
        heapq.heapify(heap)
        out = []
        while heap:
            u = heapq.heappop(heap)
            out.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(heap, v)
        ordered[p] = out

    left = [i for i in range(n) if indeg[i] > 0]
    if left:
        return ("error", "order_cycle", tuple(items[i][0] for i in left))

    eff = [None] * n
    for i in range(n):
        if eff[i] is not None:
            continue
        path = []
        u = i
        while True:
            if eff[u] is not None:
                res = eff[u]
                break
            s = srcs[u]
            if s is None:
                res = bool(items[u][3])
                path.append(u)
                break
            path.append(u)
            u = s
        for v in path:
            eff[v] = res

    filt = set(active_filter)
    adm = [False] * n
    if filt:
        for i in range(n):
            if eff[i]:
                for t in items[i][4]:
                    if t in filt:
                        adm[i] = True
                        break
    else:
        for i in range(n):
            adm[i] = eff[i]

    res = []
    stack = ordered[n][::-1]
    while stack:
        u = stack.pop()
        if not adm[u]:
            continue
        res.append(items[u][0])
        ch = ordered[u]
        for k in range(len(ch) - 1, -1, -1):
            stack.append(ch[k])
    return ("ok", tuple(res))