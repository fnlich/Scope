def warehouse_groups(campaigns, requests):
    import heapq

    camp = {c[0]: c for c in campaigns}
    stock = {c[0]: c[5] for c in campaigns}

    records = []
    groups = {}
    deadlines = []
    expiries = []
    counted = {}
    active = {}
    seq = 0

    def finish(g, status, terminal):
        cid = g[1]
        if status == "SHIPPED":
            g[6] = g[3]
            for uid, qty in g[5]:
                key = (cid, uid)
                active.pop(key, None)
                heapq.heappush(expiries, (terminal + camp[cid][8], seq_counter[0], key, qty))
                seq_counter[0] += 1
        else:
            stock[cid] += g[3]
            for uid, qty in g[5]:
                key = (cid, uid)
                active.pop(key, None)
                counted[key] = counted.get(key, 0) - qty
                if counted[key] == 0:
                    del counted[key]
        g[4] = status
        g[7] = terminal
        for i in g[8]:
            r = records[i]
            records[i] = ("ACCEPTED", "", g[0], status, g[6], terminal)

    seq_counter = [0]

    def process_deadlines(t):
        while deadlines and deadlines[0][0] <= t:
            d, gid = heapq.heappop(deadlines)
            g = groups.get(gid)
            if g is None or g[4] != "ACTIVE" or g[2] != d:
                continue
            cid = g[1]
            c = camp[cid]
            if c[9] == "VOID":
                finish(g, "VOID", d)
            elif stock[cid] >= c[4] - g[3]:
                gap = c[4] - g[3]
                stock[cid] -= gap
                g[3] = c[4]
                finish(g, "SHIPPED", d)
            else:
                finish(g, "VOID", d)

    def process_expiries(t):
        while expiries and expiries[0][0] <= t:
            _, _, key, qty = heapq.heappop(expiries)
            counted[key] = counted.get(key, 0) - qty
            if counted[key] == 0:
                del counted[key]

    for pos, req in enumerate(requests):
        t, cid, uid, qty, ref = req
        process_deadlines(t)
        process_expiries(t)

        reason = None
        c = camp.get(cid)

        if c is None:
            reason = "UNKNOWN_CAMPAIGN"
        elif qty < 1 or qty > c[6]:
            reason = "ORDER_LIMIT"
        elif (cid, uid) in active:
            reason = "ACTIVE_USER"
        elif counted.get((cid, uid), 0) + qty > c[7]:
            reason = "USER_LIMIT"
        elif ref == 0 and not (c[1] <= t < c[2]):
            reason = "CAMPAIGN_CLOSED"
        elif ref == 0 and qty > c[4]:
            reason = "GROUP_CAPACITY"
        elif ref != 0:
            g = groups.get(ref)
            if g is None or g[1] != cid:
                reason = "GROUP_MISSING"
            elif g[4] != "ACTIVE":
                reason = "GROUP_CLOSED"
            elif g[3] + qty > c[4]:
                reason = "GROUP_CAPACITY"

        if reason is None and qty > stock[cid]:
            reason = "OUT_OF_STOCK"

        if reason is not None:
            records.append(("REJECTED", reason, 0, "", 0, -1))
            continue

        key = (cid, uid)
        counted[key] = counted.get(key, 0) + qty

        if ref == 0:
            gid = pos + 1
            deadline = t + c[3]
            g = [gid, cid, deadline, qty, "ACTIVE", [(uid, qty)], None, None, [pos]]
            groups[gid] = g
            active[key] = gid
            stock[cid] -= qty
            records.append(("ACCEPTED", "", gid, "ACTIVE", qty, -1))
            heapq.heappush(deadlines, (deadline, gid))
            if qty == c[4]:
                finish(g, "SHIPPED", t)
        else:
            gid = ref
            g = groups[gid]
            g[3] += qty
            g[5].append((uid, qty))
            g[8].append(pos)
            active[key] = gid
            stock[cid] -= qty
            records.append(("ACCEPTED", "", gid, "ACTIVE", g[3], -1))
            if g[3] == c[4]:
                finish(g, "SHIPPED", t)

    while deadlines:
        d, gid = heapq.heappop(deadlines)
        g = groups.get(gid)
        if g is None or g[4] != "ACTIVE" or g[2] != d:
            continue
        cid = g[1]
        c = camp[cid]
        if c[9] == "VOID":
            finish(g, "VOID", d)
        elif stock[cid] >= c[4] - g[3]:
            gap = c[4] - g[3]
            stock[cid] -= gap
            g[3] = c[4]
            finish(g, "SHIPPED", d)
        else:
            finish(g, "VOID", d)

    return records, [stock[c[0]] for c in campaigns]