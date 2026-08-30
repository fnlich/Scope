import hashlib


def cache_audit(events):
    def encode(endpoint, payload):
        out = [b"R"]
        eb = endpoint.encode("utf-8")
        out.append(b"S%d:" % len(eb))
        out.append(eb)
        stack = [payload]
        while stack:
            v = stack.pop()
            if type(v) is bytes:
                out.append(v)
            elif v is None:
                out.append(b"N")
            elif v is True:
                out.append(b"T")
            elif v is False:
                out.append(b"F")
            elif isinstance(v, int):
                out.append(b"I%d;" % v)
            elif isinstance(v, str):
                sb = v.encode("utf-8")
                out.append(b"S%d:" % len(sb))
                out.append(sb)
            elif isinstance(v, list):
                out.append(b"A%d:" % len(v))
                for i in range(len(v) - 1, -1, -1):
                    stack.append(v[i])
            else:
                items = sorted(v.items(), key=lambda kv: kv[0].encode("utf-8"))
                out.append(b"O%d:" % len(items))
                for k, val in reversed(items):
                    stack.append(val)
                    kb = k.encode("utf-8")
                    stack.append(b"S%d:" % len(kb) + kb)
        return hashlib.sha256(b"".join(out)).digest()

    result = []
    cache = set()
    journal = []
    marks = []
    for idx in range(len(events)):
        ev = events[idx]
        op = ev[0]
        if op == "read":
            if not marks:
                result.append(idx)
            else:
                k = encode(ev[1], ev[2])
                if k in cache:
                    pass
                else:
                    cache.add(k)
                    journal.append((k, False))
                    result.append(idx)
        elif op == "invalidate":
            if marks:
                k = encode(ev[1], ev[2])
                if k in cache:
                    cache.discard(k)
                    journal.append((k, True))
        elif op == "enter":
            marks.append(len(journal))
        else:
            mark = marks.pop()
            if not marks:
                cache.clear()
                del journal[:]
            elif ev[1] == "discard":
                i = len(journal) - 1
                while i >= mark:
                    k, was = journal[i]
                    if was:
                        cache.add(k)
                    else:
                        cache.discard(k)
                    i -= 1
                del journal[mark:]
    return result