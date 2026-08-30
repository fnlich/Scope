def recover_relay(slots, packets, now):
    by_id = {}
    for p in packets:
        by_id[p["id"]] = p

    def safe(path):
        if not isinstance(path, str) or path == "":
            return False
        if "\\" in path or "\x00" in path:
            return False
        if path[0] == "/" or path[-1] == "/":
            return False
        for c in path.split("/"):
            if c == "" or c == "." or c == "..":
                return False
        return True

    def build(lst):
        d = {}
        for entry in lst:
            path = entry[0]
            size = entry[1]
            if not safe(path):
                return None
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                return None
            if path in d:
                return None
            d[path] = size
        return d

    local = {}
    for p in packets:
        pid = p["id"]
        ok = True
        if not p.get("metadata_readable"):
            ok = False
        elif p.get("sealed") is None or p.get("manifest") is None:
            ok = False
        else:
            f = build(p["files"])
            if f is None:
                ok = False
            else:
                m = build(p["manifest"])
                if m is None or m != f:
                    ok = False
                elif f.get("state.msg", 0) <= 0:
                    ok = False
        local[pid] = ok

    base = {}
    for p in packets:
        pid = p["id"]
        if not local[pid]:
            base[pid] = False
            continue
        ps = p["parents"]
        good = True
        seen = set()
        for q in ps:
            if q in seen:
                good = False
                break
            seen.add(q)
        if good:
            for q in ps:
                par = by_id.get(q)
                if par is None or not local[q]:
                    good = False
                    break
                if par["sealed"] > p["sealed"]:
                    good = False
                    break
        base[pid] = good

    state = {}
    for p in packets:
        start = p["id"]
        if start in state:
            continue
        if not base[start]:
            state[start] = 0
            continue
        stack = [(start, 0)]
        onpath = set()
        while stack:
            node, idx = stack.pop()
            if idx == 0:
                if node in state:
                    continue
                if not base[node]:
                    state[node] = 0
                    continue
                onpath.add(node)
            pk = by_id[node]
            ps = pk["parents"]
            res = 1
            found = False
            while idx < len(ps):
                q = ps[idx]
                if q in state:
                    if state[q] == 0:
                        res = 0
                        break
                    idx += 1
                    continue
                if q in onpath:
                    res = 0
                    break
                stack.append((node, idx))
                stack.append((q, 0))
                found = True
                break
            if found:
                continue
            state[node] = res
            onpath.discard(node)

    sound = state

    restore = []
    chosen = {}
    cand_by_slot = {}
    for p in packets:
        if p["displaced"] is None:
            continue
        s = p["slot"]
        if s not in cand_by_slot:
            cand_by_slot[s] = []
        cand_by_slot[s].append(p)

    for i, sl in enumerate(slots):
        if sl["active"] is not None:
            continue
        ps_ = sl["promotion_started"]
        if ps_ is not None and not (ps_ > now):
            continue
        best = None
        lo = sl["lo"]
        hi = sl["hi"]
        for p in cand_by_slot.get(i, ()):
            if sound.get(p["id"], 0) != 1:
                continue
            se = p["sealed"]
            di = p["displaced"]
            if se is None:
                continue
            if not (lo <= se and se <= di and di <= hi):
                continue
            if best is None or di > best["displaced"] or (di == best["displaced"] and p["id"] < best["id"]):
                best = p
        if best is not None:
            restore.append([i, best["id"]])
            chosen[i] = best["id"]

    keep = set()
    stack = []
    for sl in slots:
        if sl["active"] is not None:
            stack.append(sl["active"])
    for i, pid in chosen.items():
        stack.append(pid)
    visited = set()
    while stack:
        pid = stack.pop()
        if pid in visited:
            continue
        visited.add(pid)
        pk = by_id.get(pid)
        if pk is None:
            continue
        if pk["displaced"] is not None:
            keep.add(pid)
        if pk["metadata_readable"]:
            for q in pk["parents"]:
                if q in by_id and q not in visited:
                    stack.append(q)

    delete = []
    for p in packets:
        if p["displaced"] is None:
            continue
        pid = p["id"]
        if pid in keep:
            continue
        if not p["metadata_readable"]:
            continue
        delete.append(pid)
    delete.sort()
    return {"restore": restore, "delete": delete}