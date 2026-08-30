from collections import deque

BASE = 10 ** 6
PEN = 10 ** 3
INF = float('inf')


def _mcmf(edges):
    if not edges:
        return 0
    lkeys = {}
    rkeys = {}
    for (l, r, w) in edges:
        if l not in lkeys:
            lkeys[l] = len(lkeys)
        if r not in rkeys:
            rkeys[r] = len(rkeys)
    nl = len(lkeys)
    nr = len(rkeys)
    N = nl + nr + 2
    src = 0
    snk = N - 1
    to = []
    cap = []
    cst = []
    head = [[] for _ in range(N)]

    def ae(u, v, c, w):
        head[u].append(len(to))
        to.append(v)
        cap.append(c)
        cst.append(w)
        head[v].append(len(to))
        to.append(u)
        cap.append(0)
        cst.append(-w)

    for i in range(nl):
        ae(src, 1 + i, 1, 0)
    for j in range(nr):
        ae(1 + nl + j, snk, 1, 0)
    for (l, r, w) in edges:
        ae(1 + lkeys[l], 1 + nl + rkeys[r], 1, -w)

    total = 0
    while True:
        dist = [INF] * N
        pe = [-1] * N
        inq = [False] * N
        dist[src] = 0
        q = deque([src])
        inq[src] = True
        while q:
            u = q.popleft()
            inq[u] = False
            du = dist[u]
            for eid in head[u]:
                if cap[eid] > 0:
                    v = to[eid]
                    nd = du + cst[eid]
                    if nd < dist[v]:
                        dist[v] = nd
                        pe[v] = eid
                        if not inq[v]:
                            inq[v] = True
                            q.append(v)
        if dist[snk] == INF or dist[snk] >= 0:
            break
        total += dist[snk]
        v = snk
        while v != src:
            eid = pe[v]
            cap[eid] -= 1
            cap[eid ^ 1] += 1
            v = to[eid ^ 1]
    return -total


def prepare_depth_schedule(views, cached):
    n = len(views)
    assignment = [0] * n
    elig = []
    for i, v in enumerate(views):
        if v[5] and v[6] and v[2] is not None and v[3] is not None:
            elig.append(i)
    if not elig:
        return {"assignment": assignment, "textures": []}

    groups = {}
    for i in elig:
        key = (views[i][2], views[i][3], views[i][4])
        groups.setdefault(key, []).append(i)

    cgroups = {}
    for ci, c in enumerate(cached):
        key = (c[2], c[3], c[4])
        cgroups.setdefault(key, []).append(ci)

    accepted = []

    for key in sorted(groups.keys()):
        gv = groups[key]
        gc = cgroups.get(key, [])
        cands = []
        for ci in gc:
            av = cached[ci][5]
            ct = cached[ci][1]
            for p in gv:
                if av <= views[p][7]:
                    w = BASE - PEN + (1 if ct == views[p][1] else 0)
                    cands.append(((0, ci, p), w, ('c', ci), ('v', p)))
        for u in gv:
            for v2 in gv:
                if u == v2:
                    continue
                if views[u][8] <= views[v2][7]:
                    w = BASE + (1 if views[u][1] == views[v2][1] else 0)
                    cands.append(((1, u, v2), w, ('v', u), ('v', v2)))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])

        used_pred = set()
        used_succ = set()
        excluded = set()

        def build(skip_l=None, skip_r=None, skip_idx=None):
            out = []
            for idx2, (trip, w, ln, rn) in enumerate(cands):
                if idx2 in excluded or idx2 == skip_idx:
                    continue
                if ln in used_pred or rn in used_succ:
                    continue
                if ln == skip_l or rn == skip_r:
                    continue
                out.append((ln, rn, w))
            return out

        cur = _mcmf(build())

        for idx, (trip, w, ln, rn) in enumerate(cands):
            if idx in excluded:
                continue
            if ln in used_pred or rn in used_succ:
                continue
            val = w + _mcmf(build(ln, rn, idx))
            if val == cur:
                used_pred.add(ln)
                used_succ.add(rn)
                cur -= w
                accepted.append(trip)
            else:
                excluded.add(idx)

    nxt = {}
    predof = {}
    cacheof = {}
    for (k, p, s) in accepted:
        if k == 0:
            cacheof[s] = cached[p][0]
        else:
            nxt[p] = s
            predof[s] = p

    seqs = []
    for i in elig:
        if i in predof:
            continue
        path = [i]
        cur2 = i
        while cur2 in nxt:
            cur2 = nxt[cur2]
            path.append(cur2)
        seqs.append((views[i][2], views[i][3], views[i][4], path, cacheof.get(i)))

    seqs.sort(key=lambda t: (t[0], t[1], t[2], tuple(t[3])))

    textures = []
    for num, (w, h, s, path, cid) in enumerate(seqs, 1):
        for p in path:
            assignment[p] = num
        textures.append([w, h, s, "depth32float", 0.0, cid, list(path)])

    return {"assignment": assignment, "textures": textures}