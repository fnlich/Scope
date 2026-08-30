def route_requests(routes, requests, guard_count, fallback_id):
    fb = fallback_id if isinstance(fallback_id, int) and not isinstance(fallback_id, bool) else -1

    buckets = {}
    quotas = []
    for idx, r in enumerate(routes):
        pattern, priority, rt, rf, ul = r[0], r[1], r[2], r[3], r[4]
        if pattern == "/":
            segs = ()
        else:
            segs = tuple(pattern[1:].split("/")) if pattern.startswith("/") else tuple(pattern.split("/"))
        mask = 0
        for s in segs:
            mask = (mask << 1) | (1 if s == ":" else 0)
        key = tuple("\x00" if s == ":" else s for s in segs)
        tmask = 0
        for g in rt:
            tmask |= 1 << g
        fmask = 0
        for g in rf:
            fmask |= 1 << g
        quotas.append(-1 if ul is None else ul)
        buckets.setdefault(key, []).append((-priority, idx, tmask, fmask, mask))

    for k in buckets:
        buckets[k].sort()

    out = []
    for req in requests:
        path, tg = req[0], req[1]
        if path == "/":
            psegs = []
        else:
            psegs = path[1:].split("/") if path.startswith("/") else path.split("/")
        n = len(psegs)
        gmask = 0
        for g in tg:
            gmask |= 1 << g

        chosen = -1
        cmask = 0
        found_path = False
        found_guard = False

        for sub in range(1 << n):
            key = tuple("\x00" if (sub >> (n - 1 - i)) & 1 else psegs[i] for i in range(n))
            lst = buckets.get(key)
            if not lst:
                continue
            found_path = True
            for negp, idx, tmask, fmask, mask in lst:
                if (gmask & tmask) != tmask:
                    continue
                if gmask & fmask:
                    continue
                found_guard = True
                if quotas[idx] != 0:
                    if chosen == -1 or (negp, idx) < best:
                        best = (negp, idx)
                        chosen = idx
                        cmask = mask
                    break

        if chosen != -1:
            if quotas[chosen] > 0:
                quotas[chosen] -= 1
            caps = [psegs[i] for i in range(n) if (cmask >> (n - 1 - i)) & 1]
            out.append([chosen, "MATCH", caps])
        else:
            if not found_path:
                st = "PATH_MISS"
            elif not found_guard:
                st = "GUARD_MISS"
            else:
                st = "EXHAUSTED"
            out.append([fb, st, []])
    return out