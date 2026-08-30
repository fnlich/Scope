def import_robot_forest(links, joints):
    link_set = set()
    link_list = []
    for i, nm in enumerate(links):
        if nm in link_set:
            return ("ERROR", "DUPLICATE_LINK", i, nm)
        link_set.add(nm)
        link_list.append(nm)

    def resolve(ref, decl, names):
        if ref.startswith("/"):
            c = ref[1:]
            return c if c in names else None
        segs = decl.split("/")
        for k in range(len(segs) - 1, -1, -1):
            cand = "/".join(segs[:k]) + "/" + ref if k > 0 else ref
            if cand in names:
                return cand
        return None

    jnames = set()
    jlist = []
    for i, j in enumerate(joints):
        nm = j[0]
        if nm in jnames:
            return ("ERROR", "DUPLICATE_JOINT", i, nm)
        jnames.add(nm)
        jlist.append(j)

    parent_of = {}
    owner = {}
    resolved = []
    for i, j in enumerate(jlist):
        nm, pref, cref = j[0], j[1], j[2]
        p = resolve(pref, nm, link_set)
        if p is None:
            return ("ERROR", "UNKNOWN_PARENT", i, pref)
        c = resolve(cref, nm, link_set)
        if c is None:
            return ("ERROR", "UNKNOWN_CHILD", i, cref)
        if c in owner:
            return ("ERROR", "MULTIPLE_PARENTS", i, c)
        cur = p
        seen = 0
        while cur is not None:
            if cur == c:
                return ("ERROR", "CYCLE", i, nm)
            cur = parent_of.get(cur)
            seen += 1
            if seen > len(link_list) + 2:
                break
        owner[c] = i
        parent_of[c] = p
        resolved.append((p, c))

    n = len(jlist)
    src = [None] * n
    for i, j in enumerate(jlist):
        mref = j[7]
        if mref is None:
            continue
        s = resolve(mref, j[0], jnames)
        if s is None:
            return ("ERROR", "UNKNOWN_MIMIC", i, mref)
        src[i] = s

    idx_of = {}
    for i, j in enumerate(jlist):
        idx_of[j[0]] = i

    mimic_parent = {}
    for i, j in enumerate(jlist):
        if src[i] is None:
            continue
        si = idx_of[src[i]]
        k1 = jlist[i][3]
        k2 = jlist[si][3]
        if k1 not in ("revolute", "prismatic") or k2 not in ("revolute", "prismatic") or k1 != k2:
            return ("ERROR", "INVALID_MIMIC", i, jlist[i][0])
        cur = si
        bad = False
        steps = 0
        while cur is not None:
            if cur == i:
                bad = True
                break
            cur = mimic_parent.get(cur)
            steps += 1
            if steps > n + 2:
                break
        if bad:
            return ("ERROR", "MIMIC_CYCLE", i, jlist[i][0])
        mimic_parent[i] = si

    own = [None] * n
    for i, j in enumerate(jlist):
        kind, lo, up = j[3], j[4], j[5]
        if kind in ("revolute", "prismatic") and lo is not None and up is not None and lo <= up:
            own[i] = (lo, up)

    children_m = {}
    indeg = [0] * n
    for i in range(n):
        if i in mimic_parent:
            p = mimic_parent[i]
            children_m.setdefault(p, []).append(i)
            indeg[i] = 1

    import heapq
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    eff = [None] * n
    done = [False] * n
    while heap:
        i = heapq.heappop(heap)
        if i in mimic_parent:
            p = mimic_parent[i]
            m = jlist[i][8]
            off = jlist[i][9]
            sint = eff[p]
            if m == 0:
                img = (off, off)
            elif sint is None:
                img = None
            else:
                a = m * sint[0] + off
                b = m * sint[1] + off
                img = (a, b) if a <= b else (b, a)
            o = own[i]
            if img is None:
                res = o
            elif o is None:
                res = img
            else:
                lo = max(o[0], img[0])
                hi = min(o[1], img[1])
                if lo > hi:
                    return ("ERROR", "LIMIT_CONFLICT", i, jlist[i][0])
                res = (lo, hi)
            eff[i] = res
        else:
            eff[i] = own[i]
        done[i] = True
        for c in children_m.get(i, ()):
            heapq.heappush(heap, c)

    kids = {}
    for i, (p, c) in enumerate(resolved):
        kids.setdefault(p, []).append((jlist[i][0], c, i))
    for k in kids:
        kids[k].sort(key=lambda t: (t[0], t[1]))

    roots = sorted(l for l in link_list if l not in parent_of)
    out = []
    for r in roots:
        seq = []
        stack = [(r, None)]
        while stack:
            name, inc = stack.pop()
            seq.append((name, inc))
            ch = kids.get(name)
            if ch:
                for jn, cn, ji in reversed(ch):
                    j = jlist[ji]
                    drv = None
                    if src[ji] is not None:
                        drv = (src[ji], j[8], j[9])
                    stack.append((cn, (jn, j[3], eff[ji], drv, j[6])))
        out.append(tuple(seq))
    return ("OK", tuple(out))