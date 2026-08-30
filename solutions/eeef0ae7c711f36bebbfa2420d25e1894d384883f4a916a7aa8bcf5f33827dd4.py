def authorize_cache(tenants, classes, capacity, global_capacity, lifetime, margin, events):
    from collections import OrderedDict

    class MaxTree:
        __slots__ = ("n", "t")
        def __init__(self, n):
            self.n = n
            self.t = [-1] * (2 * n)
        def update(self, l, r, v):
            n = self.n
            t = self.t
            l += n
            r += n + 1
            while l < r:
                if l & 1:
                    if t[l] < v:
                        t[l] = v
                    l += 1
                if r & 1:
                    r -= 1
                    if t[r] < v:
                        t[r] = v
                l >>= 1
                r >>= 1
        def query(self, i):
            n = self.n
            t = self.t
            i += n
            res = -1
            while i >= 1:
                if t[i] > res:
                    res = t[i]
                i >>= 1
            return res

    T = tenants
    C = classes
    ft = MaxTree(T if T > 0 else 1)
    fc = MaxTree(C if C > 0 else 1)
    fone = {}

    tenant_lru = {}
    glob = OrderedDict()
    cache = {}

    out = []
    next_id = 1
    seq = 0

    for ev in events:
        kind = ev[0]
        if kind == "request":
            _, tm, tn, cl = ev
            key = (tn, cl)
            entry = cache.get(key)
            reuse = False
            if entry is not None:
                cid, cseq, issue = entry
                life = lifetime[cl - 1] - margin[cl - 1]
                if tm < issue + life:
                    if cseq > ft.query(tn - 1) and cseq > fc.query(cl - 1) and cseq > fone.get(key, -1):
                        reuse = True
            ev_t = 0
            ev_c = 0
            if reuse:
                cid = cache[key][0]
                tl = tenant_lru[tn]
                tl.move_to_end(key)
                glob.move_to_end(key)
                out.append([cid, 0, 0, 0])
            else:
                cid = next_id
                next_id += 1
                if entry is not None:
                    cache[key] = (cid, seq, tm)
                    tl = tenant_lru[tn]
                    tl.move_to_end(key)
                    glob.move_to_end(key)
                    out.append([cid, 1, 0, 0])
                else:
                    cache[key] = (cid, seq, tm)
                    tl = tenant_lru.get(tn)
                    if tl is None:
                        tl = OrderedDict()
                        tenant_lru[tn] = tl
                    tl[key] = 1
                    glob[key] = 1
                    if len(tl) > capacity[tn - 1]:
                        vk = next(iter(tl))
                        del tl[vk]
                        del glob[vk]
                        del cache[vk]
                        ev_t = vk[0]
                        ev_c = vk[1]
                    elif len(glob) > global_capacity:
                        vk = next(iter(glob))
                        del glob[vk]
                        tl2 = tenant_lru[vk[0]]
                        del tl2[vk]
                        del cache[vk]
                        ev_t = vk[0]
                        ev_c = vk[1]
                    out.append([cid, 1, ev_t, ev_c])
        elif kind == "force_tenants":
            _, tm, l, r = ev
            ft.update(l - 1, r - 1, seq)
        elif kind == "force_classes":
            _, tm, l, r = ev
            fc.update(l - 1, r - 1, seq)
        else:
            _, tm, tn, cl = ev
            k = (tn, cl)
            if fone.get(k, -1) < seq:
                fone[k] = seq
        seq += 1

    return out