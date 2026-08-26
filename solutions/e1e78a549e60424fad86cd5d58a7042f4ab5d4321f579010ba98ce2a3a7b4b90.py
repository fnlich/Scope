def unreachable_managed(object_count, parent, alias_prone, origin, bindings, ownership, conditional_ownership, exposures, permanent_roots, temporary_frames):
    R = len(parent)

    children = [[] for _ in range(R)]
    for r, p in enumerate(parent):
        if p != -1:
            children[p].append(r)

    bound = [[] for _ in range(R)]
    for r, o in bindings:
        bound[r].append(o)

    owns = [[] for _ in range(object_count)]
    for a, b in ownership:
        owns[a].append(b)

    exposed = [[] for _ in range(object_count)]
    for o, r in exposures:
        exposed[o].append(r)

    k = len(conditional_ownership)
    cond_by_obj = [[] for _ in range(object_count)]
    cond_by_reg = [[] for _ in range(R)]
    for i, (a, r, b) in enumerate(conditional_ownership):
        cond_by_obj[a].append(i)
        cond_by_reg[r].append(i)

    cond_a = [0] * k
    cond_r = [0] * k
    cond_b = [0] * k
    for i, (a, r, b) in enumerate(conditional_ownership):
        cond_a[i] = a
        cond_r[i] = r
        cond_b[i] = b

    active = [False] * R
    reachable = [False] * object_count
    region_queue = []
    object_queue = []

    for frame in temporary_frames:
        for r in frame:
            if not active[r]:
                active[r] = True
                region_queue.append(r)

    for r in permanent_roots:
        if not reachable[r]:
            reachable[r] = True
            object_queue.append(r)

    while region_queue or object_queue:
        while region_queue:
            start = region_queue.pop()
            stack = [start]
            while stack:
                r = stack.pop()

                for o in bound[r]:
                    if not reachable[o]:
                        reachable[o] = True
                        object_queue.append(o)

                if alias_prone[r]:
                    p = parent[r]
                    if p != -1 and not active[p]:
                        active[p] = True
                        region_queue.append(p)

                o = origin[r]
                if o != -1 and not active[o]:
                    active[o] = True
                    region_queue.append(o)

                for i in cond_by_reg[r]:
                    a = cond_a[i]
                    if reachable[a]:
                        b = cond_b[i]
                        if not reachable[b]:
                            reachable[b] = True
                            object_queue.append(b)

                for c in children[r]:
                    if not active[c]:
                        active[c] = True
                        stack.append(c)

        while object_queue:
            a = object_queue.pop()

            for b in owns[a]:
                if not reachable[b]:
                    reachable[b] = True
                    object_queue.append(b)

            for r in exposed[a]:
                if not active[r]:
                    active[r] = True
                    region_queue.append(r)

            for i in cond_by_obj[a]:
                r = cond_r[i]
                if active[r]:
                    b = cond_b[i]
                    if not reachable[b]:
                        reachable[b] = True
                        object_queue.append(b)

    return [i for i in range(object_count) if not reachable[i]]