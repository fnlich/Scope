def assemble_telemetry(records, aliases):
    valid_kinds = ("value", "datatype", "description")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")

    def norm(path):
        if not isinstance(path, str):
            return None
        segs = []
        for s in path.split("/"):
            if s == "" or s == ".":
                continue
            if s == "..":
                if not segs:
                    return None
                segs.pop()
                continue
            for ch in s:
                if ch not in allowed:
                    return None
            segs.append(s.lower())
        if not segs:
            return None
        return "/".join(segs)

    values = {}
    meta = {}
    for rec in records:
        try:
            key, rev, payload = rec[0], rec[1], rec[2]
        except Exception:
            continue
        if not isinstance(key, str):
            continue
        idx = key.rfind("::")
        if idx < 0:
            continue
        kind = key[idx + 2:]
        if kind not in valid_kinds:
            continue
        p = norm(key[:idx])
        if p is None:
            continue
        if kind == "value":
            d = values.setdefault(p, {})
            if rev in d:
                prev = d[rev]
                if prev is not None:
                    pv = prev[0]
                    if type(pv) is not type(payload) or not (pv == payload):
                        d[rev] = None
            else:
                d[rev] = (payload,)
        else:
            d = meta.setdefault((kind, p), {})
            if rev in d:
                prev = d[rev]
                if prev is not None:
                    pv = prev[0]
                    if type(pv) is not type(payload) or not (pv == payload):
                        d[rev] = None
            else:
                d[rev] = (payload,)

    type_map = {}
    for n in ("bool", "boolean"):
        type_map[n] = "boolean"
    for b in ("8", "16", "32", "64"):
        type_map["int" + b] = "integer"
        type_map["uint" + b] = "integer"
        type_map["sint" + b] = "integer"
    for n in ("float32", "float64", "decimal"):
        type_map[n] = "number"

    props = []
    for p, revmap in values.items():
        best = max(revmap)
        entry = revmap[best]
        if entry is None:
            continue
        val = entry[0]
        segs = p.split("/")
        ancestors = []
        cur = ""
        for s in segs:
            cur = s if not cur else cur + "/" + s
            ancestors.append(cur)
        bad = False
        chosen = {}
        for kind in ("datatype", "description"):
            sel_rev = -1
            sel_entry = None
            for depth in range(len(ancestors)):
                d = meta.get((kind, ancestors[depth]))
                if not d:
                    continue
                r = -1
                for rv in d:
                    if rv <= best and rv > r:
                        r = rv
                if r < 0:
                    continue
                if r >= sel_rev:
                    sel_rev = r
                    sel_entry = d[r]
            if sel_entry is None:
                chosen[kind] = None
            elif sel_entry is None or sel_entry == None:
                chosen[kind] = None
            else:
                chosen[kind] = sel_entry
            if sel_rev >= 0 and sel_entry is None:
                bad = True
        if bad:
            continue
        dt = chosen["datatype"]
        ds = chosen["description"]
        if dt is None:
            mapped = "string"
        else:
            raw = dt[0]
            if isinstance(raw, str):
                mapped = type_map.get(raw.strip().lower(), "string")
            else:
                mapped = "string"
        if ds is None:
            desc = ""
        else:
            desc = ds[0]
        props.append((p, val, mapped, desc))

    alias_map = {}
    conflicting = set()
    for pair in aliases:
        try:
            src, tgt = pair[0], pair[1]
        except Exception:
            continue
        s = norm(src)
        t = norm(tgt)
        if s is None or t is None:
            continue
        if s in alias_map:
            if alias_map[s] != t:
                conflicting.add(s)
        else:
            alias_map[s] = t

    resolved = {}

    def resolve(start):
        if start in resolved:
            return resolved[start]
        chain = []
        seen = set()
        cur = start
        res = None
        while True:
            if cur in conflicting:
                res = None
                break
            if cur in resolved:
                res = resolved[cur]
                break
            if cur in seen:
                res = None
                break
            seen.add(cur)
            chain.append(cur)
            nxt = alias_map.get(cur)
            if nxt is None:
                res = cur
                break
            cur = nxt
        for c in chain:
            resolved[c] = res
        return res

    groups = {}
    for p, val, mapped, desc in props:
        ident = resolve(p)
        if ident is None:
            continue
        if ident in groups:
            g = groups[ident]
            if g is not None:
                ov, om, od = g
                ok = (type(ov) is type(val) and ov == val and om == mapped
                      and type(od) is type(desc) and od == desc)
                if not ok:
                    groups[ident] = None
        else:
            groups[ident] = (val, mapped, desc)

    out = []
    for ident in sorted(groups):
        g = groups[ident]
        if g is None:
            continue
        out.append((ident, g[0], g[1], g[2]))
    return out