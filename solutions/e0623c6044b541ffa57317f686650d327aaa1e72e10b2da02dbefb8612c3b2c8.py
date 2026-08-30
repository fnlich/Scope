def audit_projections(initial_schema, changes, requests):
    n = len(initial_schema)
    gname = []
    gsets = []
    name2g = {}
    colg = [0] * n
    for i in range(n):
        nm = initial_schema[i]
        g = name2g.get(nm)
        if g is None:
            g = len(gname)
            gname.append(nm)
            gsets.append(set())
            name2g[nm] = g
        gsets[g].add(i)
        colg[i] = g

    m = len(changes)
    children = [[] for _ in range(m + 1)]
    for i in range(m):
        children[changes[i][0]].append(i + 1)

    reqs = [[] for _ in range(m + 1)]
    for ri in range(len(requests)):
        r = requests[ri]
        reqs[r[0]].append(ri)

    results = [None] * len(requests)

    def apply_change(ch):
        und = []
        kind = ch[1]
        if kind == "RENAME":
            i = ch[2]
            new = ch[3]
            g = colg[i]
            if gname[g] != new:
                gsets[g].discard(i)
                und.append((0, g, i))
                h = name2g.get(new)
                if h is None:
                    h = len(gname)
                    gname.append(new)
                    gsets.append(set())
                    name2g[new] = h
                    und.append((3, new, None))
                gsets[h].add(i)
                und.append((1, h, i))
                und.append((2, i, g))
                colg[i] = h
        else:
            a = ch[2]
            b = ch[3]
            if a != b:
                ga = name2g.get(a)
                gb = name2g.get(b)
                if ga is not None or gb is not None:
                    if ga is None:
                        ga = len(gname)
                        gname.append(a)
                        gsets.append(set())
                        name2g[a] = ga
                        und.append((3, a, None))
                    if gb is None:
                        gb = len(gname)
                        gname.append(b)
                        gsets.append(set())
                        name2g[b] = gb
                        und.append((3, b, None))
                    und.append((4, ga, a))
                    und.append((4, gb, b))
                    und.append((3, a, ga))
                    und.append((3, b, gb))
                    gname[ga] = b
                    gname[gb] = a
                    name2g[a] = gb
                    name2g[b] = ga
        return und

    def undo(und):
        for k in range(len(und) - 1, -1, -1):
            op = und[k]
            t = op[0]
            if t == 0:
                gsets[op[1]].add(op[2])
            elif t == 1:
                gsets[op[1]].discard(op[2])
            elif t == 2:
                colg[op[1]] = op[2]
            elif t == 3:
                if op[2] is None:
                    if op[1] in name2g:
                        del name2g[op[1]]
                else:
                    name2g[op[1]] = op[2]
            else:
                gname[op[1]] = op[2]

    def handle(v):
        for ri in reqs[v]:
            fields = requests[ri][1]
            out = []
            fails = []
            pos = 0
            for nm in fields:
                g = name2g.get(nm)
                if g is None or len(gsets[g]) == 0:
                    fails.append([pos, nm, "MISSING"])
                elif len(gsets[g]) > 1:
                    fails.append([pos, nm, "AMBIGUOUS"])
                else:
                    for x in gsets[g]:
                        out.append(x)
                pos += 1
            if fails:
                results[ri] = ["ERROR", fails]
            else:
                results[ri] = ["OK", out]

    stack = [(0, 0, None)]
    while stack:
        v, phase, und = stack.pop()
        if phase == 0:
            if v == 0:
                u = []
            else:
                u = apply_change(changes[v - 1])
            handle(v)
            stack.append((v, 1, u))
            for c in children[v]:
                stack.append((c, 0, None))
        else:
            undo(und)

    return results