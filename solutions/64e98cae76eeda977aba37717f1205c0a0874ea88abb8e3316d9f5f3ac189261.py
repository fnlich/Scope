def play_cache_vault(routes, capacity, commands):
    strategies = dict(routes)
    cards = {}
    schemas = {}
    groups = {}
    expiry_heap = []
    lru_heap = []
    memory_size = 0
    log = []

    import heapq

    def group_key(chest, route, base):
        return (chest, route, base)

    def card_key(chest, route, base, signature):
        return (chest, route, base, signature)

    def destroy(key):
        nonlocal memory_size
        card = cards.pop(key, None)
        if card is None:
            return False
        chest, route, base, signature = key
        g = group_key(chest, route, base)
        s = groups.get(g)
        if s is not None:
            s.discard(key)
            if not s:
                groups.pop(g, None)
                schemas.pop(g, None)
        if chest == "M":
            memory_size -= card[0]
        return True

    def destroy_group(chest, route, base):
        g = group_key(chest, route, base)
        s = groups.get(g)
        if not s:
            return 0
        keys = list(s)
        for key in keys:
            destroy(key)
        return len(keys)

    def destroy_route(route):
        for chest in ("M", "D"):
            relevant = [g for g in groups if g[0] == chest and g[1] == route]
            for _, _, base in relevant:
                destroy_group(chest, route, base)

    def make_signature(schema, request):
        req = dict(request)
        return tuple(req.get(field) for field in schema)

    def ensure_schema(chest, route, base, schema):
        g = group_key(chest, route, base)
        old = schemas.get(g)
        if old is not None and old != schema:
            destroy_group(chest, route, base)
        if g not in schemas:
            schemas[g] = schema

    def store(route, base, vary, values, size, ttl, now):
        chest = strategies[route]
        schema = tuple(sorted(vary))
        by_name = dict(zip(vary, values))
        signature = tuple(by_name[name] for name in schema)
        ensure_schema(chest, route, base, schema)
        key = card_key(chest, route, base, signature)
        old = cards.get(key)
        if old is not None:
            if chest == "M":
                nonlocal_dummy = None
            destroy(key)
        expiry = now + ttl
        cards[key] = [size, expiry, now]
        g = group_key(chest, route, base)
        groups.setdefault(g, set()).add(key)
        heapq.heappush(expiry_heap, (expiry, key))
        if chest == "M":
            nonlocal memory_size
            memory_size += size
            heapq.heappush(lru_heap, (now, key))

    def touch(key, now):
        card = cards.get(key)
        if card is not None:
            card[2] = now
            heapq.heappush(lru_heap, (now, key))

    def expire(now):
        while expiry_heap and expiry_heap[0][0] <= now:
            exp, key = heapq.heappop(expiry_heap)
            card = cards.get(key)
            if card is not None and card[1] == exp:
                destroy(key)

    def evict():
        while memory_size > capacity:
            while lru_heap:
                touched, key = heapq.heappop(lru_heap)
                card = cards.get(key)
                if card is not None and key[0] == "M" and card[2] == touched:
                    destroy(key)
                    break
            else:
                break

    for i, cmd in enumerate(commands):
        expire(i)
        op = cmd[0]

        if op == "CONFIG":
            _, route, strategy = cmd
            destroy_route(route)
            strategies[route] = strategy

        elif op == "STORE":
            _, route, base, vary, values, size, ttl = cmd
            store(route, base, vary, values, size, ttl, i)
            evict()

        elif op == "FETCH":
            _, route, base, request = cmd
            chest = strategies[route]
            g = group_key(chest, route, base)
            schema = schemas.get(g)
            if schema is None:
                log.append("MISS")
            else:
                signature = make_signature(schema, request)
                key = card_key(chest, route, base, signature)
                if key in cards:
                    log.append("HIT")
                    if chest == "M":
                        touch(key, i)
                else:
                    log.append("MISS")

        elif op == "PURGE":
            _, route, claimed, base, request = cmd
            if claimed not in ("M", "D"):
                log.append("INVALID_STRATEGY")
                continue
            if route not in strategies:
                log.append("UNKNOWN_ROUTE")
                continue
            if strategies[route] != claimed:
                log.append("STRATEGY_MISMATCH")
                continue

            if claimed == "M":
                count = destroy_group("M", route, base)
                if count:
                    log.append("DELETED " + str(count))
                else:
                    log.append("NOT_FOUND")
            else:
                g = group_key("D", route, base)
                schema = schemas.get(g)
                if schema is None:
                    log.append("NOT_FOUND")
                else:
                    signature = make_signature(schema, request)
                    key = card_key("D", route, base, signature)
                    if destroy(key):
                        log.append("DELETED 1")
                    else:
                        log.append("NOT_FOUND")

    return log