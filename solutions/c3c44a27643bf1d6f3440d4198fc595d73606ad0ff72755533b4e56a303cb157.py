def reconcile_profiles(records, operations):
    import heapq

    profiles = {}
    parents = {}
    sizes = {}
    comps = {}
    owner_heaps = {}
    current = {}
    groups = {}
    next_gid = 0

    def init_owner(owner):
        if owner not in parents:
            parents[owner] = {}
            sizes[owner] = {}
            comps[owner] = {}
            owner_heaps[owner] = []

    def ensure_node(owner, node):
        init_owner(owner)
        p = parents[owner]
        if node not in p:
            p[node] = node
            sizes[owner][node] = 1
            comps[owner][node] = {}
        return node

    def find(owner, node):
        p = parents[owner]
        x = node
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def make_group(owner, login):
        nonlocal next_gid
        g = {
            "uid": next_gid,
            "owner": owner,
            "login": login,
            "members": set(),
            "heap": [],
            "canon_heap": [],
            "canonical": None,
            "version": 0,
            "active": True
        }
        next_gid += 1
        groups[g["uid"]] = g
        return g

    def group_candidate(g, attempted=None, keep=False):
        h = g["heap"]
        temp = []
        result = None
        while h:
            e = heapq.heappop(h)
            pid = e[1]
            p = profiles.get(pid)
            if p is None or not p[7] or p[9] is not g or p[8] != e[2]:
                continue
            if pid == g["canonical"]:
                temp.append(e)
                continue
            if attempted is not None and pid in attempted:
                continue
            if p[6] >= len(p[5]):
                continue
            result = e
            break
        for e in temp:
            heapq.heappush(h, e)
        if result is not None and keep:
            heapq.heappush(h, result)
        return result

    def update_group(g):
        if not g["active"] or not g["members"]:
            return
        owner = g["owner"]
        cur = current.get(owner)
        if cur in g["members"]:
            g["canonical"] = cur
        else:
            ch = g["canon_heap"]
            canon = None
            while ch:
                created, pid, token = ch[0]
                p = profiles.get(pid)
                if p is not None and p[7] and p[9] is g and p[8] == token:
                    canon = pid
                    break
                heapq.heappop(ch)
            g["canonical"] = canon
        g["version"] += 1
        e = group_candidate(g, None, False)
        if e is not None:
            heapq.heappush(g["heap"], e)