import sys


def audit_schema_versions(parents, actions):
    sys.setrecursionlimit(10000)
    n = len(parents)
    keyset = set()
    for acts in actions:
        for a in acts:
            if a[0] != "custom_policy":
                keyset.add((a[1], a[2]))
    keys = sorted(keyset)
    m = len(keys)
    idx = {}
    pcount = 0
    for i, k in enumerate(keys):
        idx[k] = i
        if k[0] == "P":
            pcount += 1
    size = 1
    while size < max(m, 1):
        size *= 2
    tC = [0] * (2 * size)
    tE = [0] * (2 * size)

    dtype = [None] * m
    reqset = [None] * m
    cnt = [0] * m
    mins = [None] * m
    for i in range(m):
        reqset[i] = set()
        mins[i] = [(None, None)]
    curC = [0] * m
    curE = [0] * m
    pol = {"P": "compatible", "V": "compatible"}

    def upd(tree, i, delta):
        i += size
        while i >= 1:
            tree[i] += delta
            i >>= 1

    def refresh(k):
        c = cnt[k]
        d = dtype[k]
        if d is None:
            nc = c
            ne = c
        else:
            hit = 1 if d in reqset[k] else 0
            ne = c - hit
            if d.startswith("custom:"):
                nc = 0
            else:
                nc = c - hit
        if nc != curC[k]:
            upd(tC, k, nc - curC[k])
            curC[k] = nc
        if ne != curE[k]:
            upd(tE, k, ne - curE[k])
            curE[k] = ne

    def rsum(tree, l, r):
        if l > r:
            return 0
        res = 0
        l += size
        r += size + 1
        while l < r:
            if l & 1:
                res += tree[l]
                l += 1
            if r & 1:
                r -= 1
                res += tree[r]
            l >>= 1
            r >>= 1
        return res

    def find_first(tree, l, r):
        if l > r or tree[1] == 0:
            return -1

        def go(node, nl, nr):
            if nr < l or nl > r or tree[node] == 0:
                return -1
            if nl == nr:
                return nl
            mid = (nl + nr) // 2
            res = go(2 * node, nl, mid)
            if res != -1:
                return res
            return go(2 * node + 1, mid + 1, nr)

        return go(1, 0, size - 1)

    def push_min(k, t):
        m1, m2 = mins[k][-1]
        if m1 is None:
            mins[k].append((t, None))
        elif t < m1:
            mins[k].append((t, m1))
        elif m2 is None or t < m2:
            mins[k].append((m1, t))
        else:
            mins[k].append((m1, m2))

    def smallest_type(k):
        m1, m2 = mins[k][-1]
        d = dtype[k]
        s = None
        if d is not None:
            if d.startswith("custom:") and pol[keys[k][0]] == "compatible":
                s = None
            elif d in reqset[k]:
                s = d
        if s is not None and m1 == s:
            return m2
        return m1

    children = [[] for _ in range(n)]
    for i in range(1, n):
        children[parents[i]].append(i)

    results = [None] * n
    logs = [None] * n

    def apply_node(node):
        log = []
        for a in actions[node]:
            op = a[0]
            if op == "require":
                k = idx[(a[1], a[2])]
                t = a[3]
                if t in reqset[k]:
                    continue
                reqset[k].add(t)
                cnt[k] += 1
                push_min(k, t)
                refresh(k)
                log.append(("q", k, t))
            elif op == "field":
                k = idx[(a[1], a[2])]
                meta = a[3]
                if isinstance(meta, str):
                    t = meta
                else:
                    t = meta[0]
                old = dtype[k]
                dtype[k] = t
                refresh(k)
                log.append(("d", k, old))
            elif op == "remove":
                k = idx[(a[1], a[2])]
                old = dtype[k]
                if old is None:
                    continue
                dtype[k] = None
                refresh(k)
                log.append(("d", k, old))
            else:
                area = a[1]
                oldp = pol[area]
                pol[area] = a[2]
                log.append(("p", area, oldp))
        return log

    def undo_node(log):
        for e in reversed(log):
            kind = e[0]
            if kind == "q":
                k = e[1]
                reqset[k].discard(e[2])
                cnt[k] -= 1
                mins[k].pop()
                refresh(k)
            elif kind == "d":
                k = e[1]
                dtype[k] = e[2]
                refresh(k)
            else:
                pol[e[1]] = e[2]

    def compute():
        treeP = tC if pol["P"] == "compatible" else tE
        treeV = tC if pol["V"] == "compatible" else tE
        total = rsum(treeP, 0, pcount - 1) + rsum(treeV, pcount, m - 1)
        f = find_first(treeP, 0, pcount - 1)
        if f == -1:
            f = find_first(treeV, pcount, m - 1)
        if f == -1:
            return (total, None)
        area, name = keys[f]
        return (total, (area, name, smallest_type(f)))

    stack = [(0, 0)]
    while stack:
        node, state = stack.pop()
        if state == 0:
            logs[node] = apply_node(node)
            results[node] = compute()
            stack.append((node, 1))
            for c in reversed(children[node]):
                stack.append((c, 0))
        else:
            undo_node(logs[node])
            logs[node] = None

    return results