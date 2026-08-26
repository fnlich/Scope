def audit_filter_groups(schema, groups, links):
    n = len(schema)
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

    roots = [find(i) for i in range(n)]
    kinds = [None] * n
    nullable = [True] * n
    int_lo = [None] * n
    int_hi = [None] * n
    enum_dom = [None] * n

    for i, d in enumerate(schema):
        r = roots[i]
        kind = d[0]
        kinds[r] = kind
        if not d[1]:
            nullable[r] = False
        if kind == "INT":
            lo, hi = d[2], d[3]
            if int_lo[r] is None:
                int_lo[r] = lo
                int_hi[r] = hi
            else:
                if lo > int_lo[r]:
                    int_lo[r] = lo
                if hi < int_hi[r]:
                    int_hi[r] = hi
        else:
            vals = set(d[2])
            if enum_dom[r] is None:
                enum_dom[r] = vals
            else:
                enum_dom[r].intersection_update(vals)

    for r in range(n):
        if kinds[r] == "INT":
            if int_lo[r] is None:
                int_lo[r] = 0
                int_hi[r] = -1
        elif kinds[r] == "ENUM":
            if enum_dom[r] is None:
                enum_dom[r] = set()

    schema_ok = True
    for r in range(n):
        if kinds[r] == "INT":
            nonnull_ok = int_lo[r] <= int_hi[r]
        else:
            nonnull_ok = bool(enum_dom[r])
        if not nullable[r] and not nonnull_ok:
            schema_ok = False
            break

    if not schema_ok:
        return []

    import bisect

    answer = []

    for gi, group in enumerate(groups):
        acc = {}
        possible = True

        for p in group:
            c = p[0]
            op = p[1]
            r = roots[c]
            kind = kinds[r]

            a = acc.get(r)
            if a is None:
                if kind == "INT":
                    a = [0, int_lo[r], int_hi[r], None, []]
                else:
                    a = [0, None, set()]
                acc[r] = a

            if op == "IS_NULL":
                if a[0] == 2:
                    possible = False
                    break
                if not nullable[r]:
                    possible = False
                    break
                a[0] = 1
                continue

            if a[0] == 1:
                possible = False
                break
            a[0] = 2

            if op == "IS_NOT_NULL":
                continue

            if kind == "INT":
                if op == "EQ":
                    v = p[2]
                    if a[3] is None:
                        a[3] = {v}
                    else:
                        a[3].intersection_update((v,))
                elif op == "NE":
                    a[4].append((p[2], p[2]))
                elif op == "IN":
                    vals = set(p[2])
                    if a[3] is None:
                        a[3] = vals
                    else:
                        a[3].intersection_update(vals)
                elif op == "NOT_IN":
                    a[4].extend((v, v) for v in p[2])
                elif op == "LT":
                    v = p[2]
                    if v - 1 < a[2]:
                        a[2] = v - 1
                elif op == "LE":
                    v = p[2]
                    if v < a[2]:
                        a[2] = v
                elif op == "GT":
                    v = p[2]
                    if v + 1 > a[1]:
                        a[1] = v + 1
                elif op == "GE":
                    v = p[2]
                    if v > a[1]:
                        a[1] = v
                elif op == "BETWEEN":
                    lo, hi = p[2], p[3]
                    if lo > hi:
                        a[1] = 1
                        a[2] = 0
                    else:
                        if lo > a[1]:
                            a[1] = lo
                        if hi < a[2]:
                            a[2] = hi
                elif op == "NOT_BETWEEN":
                    lo, hi = p[2], p[3]
                    if lo <= hi:
                        a[4].append((lo, hi))
            else:
                if op == "EQ":
                    v = p[2]
                    if a[1] is None:
                        a[1] = {v}
                    else:
                        a[1].intersection_update((v,))
                elif op == "IN":
                    vals = set(p[2])
                    if a[1] is None:
                        a[1] = vals
                    else:
                        a[1].intersection_update(vals)
                elif op == "NE":
                    a[2].add(p[2])
                elif op == "NOT_IN":
                    a[2].update(p[2])

        if not possible:
            continue

        for r, a in acc.items():
            state = a[0]

            if state == 1:
                if not nullable[r]:
                    possible = False
                    break
                continue

            if kinds[r] == "INT":
                lo, hi, pos, neg = a[1], a[2], a[3], a[4]
                if lo > hi:
                    possible = False
                    break

                if pos is not None:
                    if not pos:
                        possible = False
                        break

                    if neg:
                        neg.sort()
                        merged = []
                        cur_a, cur_b = neg[0]
                        for x, y in neg[1:]:
                            if x <= cur_b + 1:
                                if y > cur_b:
                                    cur_b = y
                            else:
                                merged.append((cur_a, cur_b))
                                cur_a, cur_b = x, y
                        merged.append((cur_a, cur_b))
                        starts = [x[0] for x in merged]

                        found = False
                        for v in pos:
                            if v < lo or v > hi:
                                continue
                            j = bisect.bisect_right(starts, v) - 1
                            if j < 0 or v > merged[j][1]:
                                found = True
                                break
                        if not found:
                            possible = False
                            break
                    else:
                        if not any(lo <= v <= hi for v in pos):
                            possible = False
                            break
                elif neg:
                    neg.sort()
                    cur = lo
                    covered = False
                    for x, y in neg:
                        if y < cur:
                            continue
                        if x > hi:
                            break
                        if x > cur:
                            break
                        if y >= cur:
                            cur = y + 1
                            if cur > hi:
                                covered = True
                                break
                    if covered:
                        possible = False
                        break
            else:
                dom = enum_dom[r]
                pos = a[1]
                neg = a[2]

                if not dom:
                    possible = False
                    break

                if pos is None:
                    if len(dom) <= len(neg):
                        if not any(v not in neg for v in dom):
                            possible = False
                            break
                else:
                    if len(pos) <= len(dom):
                        if not any(v in dom and v not in neg for v in pos):
                            possible = False
                            break
                    else:
                        if not any(v in pos and v not in neg for v in dom):
                            possible = False
                            break

        if possible:
            answer.append(gi)

    return answer