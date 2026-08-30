def audit_dispatch_abi(feature_count, prerequisites, releases):
    cl = [(1 << f) | (prerequisites[f] if f < len(prerequisites) else 0) for f in range(feature_count)]
    changed = True
    while changed:
        changed = False
        for f in range(feature_count):
            m = cl[f]
            nm = m
            x = m
            while x:
                b = x & -x
                g = b.bit_length() - 1
                if g < feature_count:
                    nm |= cl[g]
                x ^= b
            if nm != m:
                cl[f] = nm
                changed = True

    def exposed(snap, mask):
        out = []
        for e in snap:
            g = e[2]
            if g == -1 or ((mask >> g) & 1):
                out.append((e[0], e[1]))
        return out

    for i in range(1, len(releases)):
        prev = releases[i - 1]
        cur = releases[i]
        sbits = 0
        for e in prev:
            if e[2] != -1:
                sbits |= 1 << e[2]
        for e in cur:
            if e[2] != -1:
                sbits |= 1 << e[2]
        feats = []
        x = sbits
        while x:
            b = x & -x
            feats.append(b.bit_length() - 1)
            x ^= b
        k = len(feats)
        best = None
        for sub in range(1 << k):
            pm = 0
            mask = 0
            j = 0
            t = sub
            while t:
                if t & 1:
                    f = feats[j]
                    pm |= 1 << f
                    mask |= cl[f]
                t >>= 1
                j += 1
            if (mask & sbits) != pm:
                continue
            if best is not None and mask >= best[0]:
                continue
            pl = exposed(prev, mask)
            cl2 = exposed(cur, mask)
            pos = -1
            lp = len(pl)
            lc = len(cl2)
            n = lp if lp < lc else lc
            idx = 0
            while idx < n:
                if pl[idx] != cl2[idx]:
                    pos = idx
                    break
                idx += 1
            if pos == -1:
                if lp > lc:
                    pos = lc
                else:
                    continue
            expected = pl[pos]
            actual = cl2[pos] if pos < lc else None
            if actual is None:
                kind = "deletion"
            elif actual[0] == expected[0]:
                kind = "signature"
            else:
                curnames = set()
                for e in cl2:
                    curnames.add(e[0])
                if expected[0] not in curnames:
                    kind = "deletion"
                else:
                    prevnames = set()
                    for e in pl:
                        prevnames.add(e[0])
                    if actual[0] not in prevnames:
                        kind = "insertion"
                    else:
                        kind = "reorder"
            best = (mask, pos, kind, expected, actual)
        if best is not None:
            return ("incompatible", i, best[0], best[1], best[2], best[3], best[4])
    return ("ok",)