def resolve_markers(shape_parent, layout_parent, master_parent, shape_rules, layout_rules, master_rules, paragraphs):
    from array import array
    from bisect import bisect_right

    m = len(paragraphs)
    result = [None] * m
    levels = [p[5] for p in paragraphs]
    qorder = sorted(range(m), key=levels.__getitem__)

    for i, p in enumerate(paragraphs):
        d = p[0]
        if d != "?":
            result[i] = [d, "D"]

    def make_euler(parent):
        n = len(parent)
        if n == 0:
            return array('i'), array('i'), array('i')

        head = array('i', [-1]) * n
        nxt = array('i', [-1]) * n
        depth = array('i', [0]) * n

        roots = []
        for i in range(n):
            p = parent[i]
            if p < 0:
                roots.append(i)
            else:
                depth[i] = depth[p] + 1
                nxt[i] = head[p]
                head[p] = i

        tin = array('i', [0]) * n
        order = array('i')
        stack = roots[:]

        while stack:
            v = stack.pop()
            tin[v] = len(order)
            order.append(v)
            c = head[v]
            while c != -1:
                stack.append(c)
                c = nxt[c]

        size = array('i', [1]) * n
        for k in range(len(order) - 1, -1, -1):
            v = order[k]
            p = parent[v]
            if p >= 0:
                size[p] += size[v]

        return tin, size, depth

    def range_apply(seg, base, l, r, v, depth):
        while l < r:
            if l & 1:
                old = seg[l]
                if old == -1 or depth[v] > depth[old]:
                    seg[l] = v
                l += 1
            if r & 1:
                r -= 1
                old = seg[r]
                if old == -1 or depth[v] > depth[old]:
                    seg[r] = v
            l >>= 1
            r >>= 1

    def point_get(seg, base, pos, depth):
        p = base + pos
        best = -1
        while p:
            v = seg[p]
            if v != -1 and (best == -1 or depth[v] > depth[best]):
                best = v
            p >>= 1
        return best

    def apply_shape(parent, rules):
        if not rules:
            return

        n = len(parent)
        if n == 0:
            return

        by_node = {}
        for node, level, marker in rules:
            by_node.setdefault(node, []).append((level, marker))

        for lst in by_node.values():
            lst.sort()

        tin, size, depth = make_euler(parent)
        base = 1
        while base < n:
            base <<= 1
        seg = array('i', [-1]) * (base << 1)

        activations = [(lst[0][0], node) for node, lst in by_node.items()]
        activations.sort()

        ap = 0
        an = len(activations)

        for qi in qorder:
            if result[qi] is not None:
                continue
            node = paragraphs[qi][1]
            if node < 0 or node >= n:
                continue

            level = levels[qi]
            while ap < an and activations[ap][0] <= level:
                node2 = activations[ap][1]
                range_apply(
                    seg,
                    base,
                    base + tin[node2],
                    base + tin[node2] + size[node2],
                    node2,
                    depth
                )
                ap += 1

            selected = point_get(seg, base, tin[node], depth)
            if selected == -1:
                continue

            lst = by_node[selected]
            j = bisect_right(lst, (level, "\U0010ffff")) - 1
            if j >= 0:
                result[qi] = [lst[j][1], "S"]

    def apply_layout(parent, rules, column, origin):
        if not rules:
            return

        n = len(parent)
        if n == 0:
            return

        exact = [{}, {}, {}]
        wild = {}

        role_id = {"title": 0, "body": 1, "other": 2}

        for node, role, level, marker in rules:
            if role == "*":
                wild.setdefault(node, []).append((level, marker))
            else:
                exact[role_id[role]].setdefault(node, []).append((level, marker))

        for d in exact:
            for lst in d.values():
                lst.sort()
        for lst in wild.values():
            lst.sort()

        tin, size, depth = make_euler(parent)
        base = 1
        while base < n:
            base <<= 1

        for rid in range(3):
            nodes = set(exact[rid])
            nodes.update(wild)
            if not nodes:
                continue

            activations = []
            ed = exact[rid]
            for node in nodes:
                el = ed.get(node)
                wl = wild.get(node)
                if el is None:
                    mn = wl[0][0]
                elif wl is None:
                    mn = el[0][0]
                else:
                    mn = el[0][0] if el[0][0] <= wl[0][0] else wl[0][0]
                activations.append((mn, node))
            activations.sort()

            seg = array('i', [-1]) * (base << 1)
            ap = 0
            an = len(activations)

            for qi in qorder:
                if result[qi] is not None:
                    continue
                p = paragraphs[qi]
                if p[column] < 0 or p[column] >= n:
                    continue
                if role_id[p[4]] != rid:
                    continue

                level = levels[qi]

                while ap < an and activations[ap][0] <= level:
                    node2 = activations[ap][1]
                    range_apply(
                        seg,
                        base,
                        base + tin[node2],
                        base + tin[node2] + size[node2],
                        node2,
                        depth
                    )
                    ap += 1

                node = point_get(seg, base, tin[p[column]], depth)
                if node == -1:
                    continue

                el = ed.get(node)
                wl = wild.get(node)

                ec = None
                wc = None

                if el is not None:
                    j = bisect_right(el, (level, "\U0010ffff")) - 1
                    if j >= 0:
                        ec = el[j]

                if wl is not None:
                    j = bisect_right(wl, (level, "\U0010ffff")) - 1
                    if j >= 0:
                        wc = wl[j]

                if ec is None and wc is None:
                    continue
                if ec is None:
                    chosen = wc
                elif wc is None:
                    chosen = ec
                elif ec[0] >= wc[0]:
                    chosen = ec
                else:
                    chosen = wc

                result[qi] = [chosen[1], origin]

    apply_shape(shape_parent, shape_rules)
    apply_layout(layout_parent, layout_rules, 2, "L")
    apply_layout(master_parent, master_rules, 3, "M")

    for i in range(m):
        if result[i] is None:
            p = paragraphs[i]
            if p[4] == "body" and p[5] > 0:
                result[i] = ["B", "H"]
            else:
                result[i] = ["P", "U"]

    return result