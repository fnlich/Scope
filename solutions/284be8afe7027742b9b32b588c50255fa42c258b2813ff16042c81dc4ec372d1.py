import sys


def route_diagnostics(routes, request):
    sys.setrecursionlimit(10000)
    i = request.find('?')
    if i == -1:
        path = request
        qs = None
    else:
        path = request[:i]
        qs = request[i + 1:]
    if not path.startswith('/'):
        return {"ok": False, "error": "MALFORMED_PATH"}
    if path == '/':
        segs = []
    else:
        segs = path[1:].split('/')
        for s in segs:
            if s == '':
                return {"ok": False, "error": "MALFORMED_PATH"}
    query = {}
    if qs:
        for frag in qs.split('&'):
            j = frag.find('=')
            if j <= 0:
                continue
            query[frag[:j]] = frag[j + 1:]

    def better(a, b):
        if a is None:
            return b
        if b is None:
            return a
        if b[0] > a[0]:
            return b
        if b[0] == a[0] and b[1] < a[1]:
            return b
        return a

    def solve(tokens, constraints):
        n = len(tokens)
        m = len(segs)
        memo = {}

        def go(ti, si):
            key = (ti, si)
            if key in memo:
                return memo[key]
            if ti == n:
                r = ([], []) if si == m else None
            else:
                t = tokens[ti]
                r = None
                if t.startswith('{'):
                    if len(t) > 1 and t[1] == '*':
                        name = t[2:-1]
                        c = constraints.get(name)
                        maxk = m - si
                        if c is None:
                            cands = range(1, maxk + 1)
                        else:
                            k = len(c.split('/'))
                            cands = [k] if 1 <= k <= maxk else []
                        for k in cands:
                            val = '/'.join(segs[si:si + k])
                            if c is not None and val != c:
                                continue
                            sub = go(ti + 1, si + k)
                            if sub is None:
                                continue
                            r = better(r, ([0] * k + sub[0], [val] + sub[1]))
                    else:
                        name = t[1:-1]
                        if si < m:
                            val = segs[si]
                            c = constraints.get(name)
                            if c is None or c == val:
                                sub = go(ti + 1, si + 1)
                                if sub is not None:
                                    r = better(r, ([1] + sub[0], [val] + sub[1]))
                else:
                    if si < m and segs[si] == t:
                        sub = go(ti + 1, si + 1)
                        if sub is not None:
                            r = ([2] + sub[0], sub[1])
            memo[key] = r
            return r

        return go(0, 0)

    best_spec = None
    best_id = None
    best_caps = None
    ambiguous = False

    for rt in routes:
        req = rt.get("requires") or {}
        bad = False
        for k, v in req.items():
            if k not in query or query[k] != v:
                bad = True
                break
        if bad:
            continue
        binds = rt.get("binds") or {}
        constraints = {}
        for qn, cn in binds.items():
            if qn not in query:
                bad = True
                break
            v = query[qn]
            if cn in constraints and constraints[cn] != v:
                bad = True
                break
            constraints[cn] = v
        if bad:
            continue
        pat = rt["pattern"]
        tokens = [] if pat == '/' else pat[1:].split('/')
        names = []
        for t in tokens:
            if t.startswith('{'):
                if len(t) > 1 and t[1] == '*':
                    names.append(t[2:-1])
                else:
                    names.append(t[1:-1])
        res = solve(tokens, constraints)
        if res is None:
            continue
        ranks, caps = res
        spec = ranks + [len(req) + len(binds)]
        capmap = {}
        for idx, nm in enumerate(names):
            capmap[nm] = caps[idx]
        if best_spec is None or spec > best_spec:
            best_spec = spec
            best_id = rt["id"]
            best_caps = capmap
            ambiguous = False
        elif spec == best_spec:
            ambiguous = True

    if best_spec is None:
        return {"ok": False, "error": "NO_ROUTE"}
    if ambiguous:
        return {"ok": False, "error": "AMBIGUOUS"}
    return {"ok": True, "route": best_id, "captures": best_caps, "query": query}