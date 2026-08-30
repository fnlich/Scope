def resolve_launch(records, roots):
    items = []
    errors = []

    def merge(meta_stack):
        out = {}
        for m in meta_stack:
            if m:
                out.update(m)
        return out

    for root in roots:
        stack = [("enter", root, ())]
        active = []
        stopped = [False]

        def pruned():
            for fr in active:
                if fr["limit"] is not None and fr["count"] >= fr["limit"]:
                    return True
            return False

        work = [("node", root, ())]
        while work:
            op = work.pop()
            t = op[0]
            if t == "pop":
                active.pop()
                continue
            if pruned():
                continue
            if t == "node":
                _, rid, metas = op
                rec = records[rid]
                k = rec["kind"]
                if k == "media":
                    m = merge(list(metas) + [rec.get("meta") or {}])
                    lineage = [{"id": f["id"], "index": f["index"], "count": f["count_entries"]} for f in active]
                    items.append({"id": rid, "meta": m, "lineage": lineage})
                    for f in active:
                        f["count"] += 1
                elif k == "transparent":
                    work.append(("node", rec["target"], metas + (rec.get("meta") or {},)))
                elif k == "jump":
                    work.append(("node", rec["target"], ()))
                else:
                    lineage = [{"id": f["id"], "index": f["index"], "count": f["count_entries"]} for f in active]
                    if any(f["id"] == rid for f in active):
                        idx = 0
                        for i, f in enumerate(active):
                            if f["id"] == rid:
                                idx = i
                                break
                        trail = [f["id"] for f in active[idx:]] + [rid]
                        errors.append({"kind": "cycle", "playlist": rid, "trail": trail, "lineage": lineage})
                        continue
                    limit = rec.get("limit")
                    entries = rec.get("entries") or []
                    if limit is not None and limit <= 0:
                        continue
                    if not entries:
                        continue
                    frame = {"id": rid, "limit": limit, "count": 0, "index": 0,
                             "count_entries": len(entries), "entries": entries,
                             "metas": metas + (rec.get("meta") or {},), "pos": 0}
                    active.append(frame)
                    work.append(("pop",))
                    work.append(("entry", frame))
            elif t == "entry":
                frame = op[1]
                pos = frame["pos"]
                if pos >= len(frame["entries"]):
                    continue
                frame["pos"] = pos + 1
                frame["index"] = pos + 1
                e = frame["entries"][pos]
                work.append(("entry", frame))
                work.append(("node", e["target"], frame["metas"] + (e.get("meta") or {},)))

    return {"items": items, "errors": errors}