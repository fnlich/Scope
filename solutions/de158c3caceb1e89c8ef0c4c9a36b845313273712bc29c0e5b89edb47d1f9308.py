from bisect import bisect_left, bisect_right

BIG = 1 << 60


def _build_group(items):
    cs = set()
    for f, l, ix in items:
        cs.add(f)
        cs.add(l + 1)
    c = sorted(cs)
    m = len(c) - 1
    A = [BIG] * m
    nxt = list(range(m + 2))

    def find(x):
        while nxt[x] != x:
            nxt[x] = nxt[nxt[x]]
            x = nxt[x]
        return x

    for f, l, ix in items:
        s = bisect_left(c, f)
        e = bisect_left(c, l + 1) - 1
        t = find(s)
        while t <= e:
            A[t] = ix
            nxt[t] = t + 1
            t = find(t + 1)
    size = 1
    while size < m:
        size *= 2
    tree = [BIG] * (2 * size)
    for i in range(m):
        tree[size + i] = A[i]
    for i in range(size - 1, 0, -1):
        a = tree[2 * i]
        b = tree[2 * i + 1]
        tree[i] = a if a < b else b
    return (c, tree, size, m)


def _query_group(st, a, b):
    c, tree, size, m = st
    if m <= 0:
        return BIG
    t0 = bisect_right(c, a) - 1
    if t0 < 0:
        t0 = 0
    t1 = bisect_right(c, b) - 1
    if t1 > m - 1:
        t1 = m - 1
    if t1 < t0:
        return BIG
    res = BIG
    lo = t0 + size
    hi = t1 + size + 1
    while lo < hi:
        if lo & 1:
            v = tree[lo]
            if v < res:
                res = v
            lo += 1
        if hi & 1:
            hi -= 1
            v = tree[hi]
            if v < res:
                res = v
        lo >>= 1
        hi >>= 1
    return res


def validate_schema(declarations, reserved):
    exact_raw = {}
    wild_raw = {}
    for ridx, r in enumerate(reserved):
        pat = r[0]
        f = r[1]
        l = r[2]
        if pat.endswith("::*"):
            key = pat[:-3]
            if key in wild_raw:
                wild_raw[key].append((f, l, ridx))
            else:
                wild_raw[key] = [(f, l, ridx)]
        else:
            if pat in exact_raw:
                exact_raw[pat].append((f, l, ridx))
            else:
                exact_raw[pat] = [(f, l, ridx)]

    exact = {}
    for k, v in exact_raw.items():
        exact[k] = _build_group(v)
    wild = {}
    for k, v in wild_raw.items():
        wild[k] = _build_group(v)

    decl_map = {}
    dist_firsts = []
    dist_lasts = []
    dist_idx = []

    for i in range(len(declarations)):
        d = declarations[i]
        kind = d[0]
        name = d[1]
        f = d[2]
        l = d[3]
        dist = d[4]

        if kind == "property":
            ids = [name]
        else:
            ids = [name, name + "::payload"]

        if dist and dist_firsts:
            hi = bisect_right(dist_firsts, l) - 1
            lo = bisect_left(dist_lasts, f)
            if lo <= hi:
                best = dist_idx[lo]
                for t in range(lo + 1, hi + 1):
                    if dist_idx[t] < best:
                        best = dist_idx[t]
                return (i, "MULTIPLE_DISTINGUISHED", best, "")

        best_res = None
        best_decl = None

        for idn in ids:
            r = BIG
            st = exact.get(idn)
            if st is not None:
                q = _query_group(st, f, l)
                if q < r:
                    r = q
            n = len(idn)
            p = idn.find("::")
            while p != -1:
                if p > 0 and p + 2 < n:
                    stw = wild.get(idn[:p])
                    if stw is not None:
                        q = _query_group(stw, f, l)
                        if q < r:
                            r = q
                p = idn.find("::", p + 1)
            if r < BIG:
                cand = (r, idn)
                if best_res is None or cand < best_res:
                    best_res = cand

            g = decl_map.get(idn)
            if g is not None:
                fs = g[0]
                its = g[1]
                pos = bisect_right(fs, l) - 1
                while pos >= 0:
                    it = its[pos]
                    if it[1] >= f:
                        cand = (it[2], idn)
                        if best_decl is None or cand < best_decl:
                            best_decl = cand
                        break
                    pos -= 1

        if best_res is not None:
            return (i, "USER_RESERVED", best_res[0], best_res[1])
        if best_decl is not None:
            return (i, "USER_USER", best_decl[0], best_decl[1])

        for idn in ids:
            g = decl_map.get(idn)
            if g is None:
                g = ([], [])
                decl_map[idn] = g
            fs = g[0]
            its = g[1]
            pos = bisect_right(fs, f)
            fs.insert(pos, f)
            its.insert(pos, (f, l, i))

        if dist:
            pos = bisect_right(dist_firsts, f)
            dist_firsts.insert(pos, f)
            dist_lasts.insert(pos, l)
            dist_idx.insert(pos, i)

    return None