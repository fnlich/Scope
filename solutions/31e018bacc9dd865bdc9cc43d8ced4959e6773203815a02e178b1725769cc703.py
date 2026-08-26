def process_transaction(schemas, records, subject_tenant, permissions, privileged, actions):
    from bisect import bisect_left, bisect_right

    schema_names = list(schemas.keys())
    schema_index = {name: i for i, name in enumerate(schema_names)}
    ns = len(schema_names)

    perm_masks = [0] * ns
    op_bits = {"read": 1, "add": 2, "delete": 4, "move": 8}
    for schema, op in permissions:
        si = schema_index[schema]
        perm_masks[si] |= op_bits[op]

    initial_state = [[r[2], r[3]] for r in records]
    n = len(records)

    groups = [[] for _ in range(ns)]
    for oi, r in enumerate(records):
        groups[schema_index[r[1]]].append(oi)

    ids = [0] * n
    tenants = [None] * n
    vals = [0] * n
    orig_at = [0] * n
    schema_of = [0] * n
    bases = [0] * ns
    ends = [0] * ns
    schema_ids = [[] for _ in range(ns)]
    schema_params = [None] * ns
    id_to_pos = {}

    pos = 0
    for si, name in enumerate(schema_names):
        g = groups[si]
        g.sort(key=lambda x: records[x][0])
        bases[si] = pos
        low, high, quantum = schemas[name]
        schema_params[si] = (low, high, quantum)
        local_ids = []
        for oi in g:
            r = records[oi]
            ids[pos] = r[0]
            tenants[pos] = r[2]
            vals[pos] = r[3]
            orig_at[pos] = oi
            schema_of[pos] = si
            local_ids.append(r[0])
            id_to_pos[r[0]] = pos
            pos += 1
        schema_ids[si] = local_ids
        ends[si] = pos

    live = [True] * n

    size = 4 * n + 5
    cnt = [0] * size
    sm = [0] * size
    mn = [0] * size
    mx = [0] * size
    lazy = [0] * size
    inf = 10**30

    def build(v, l, r):
        if r - l == 1:
            if tenants[l] == subject_tenant:
                cnt[v] = 1
                sm[v] = vals[l]
                mn[v] = vals[l]
                mx[v] = vals[l]
            else:
                mn[v] = inf
                mx[v] = -inf
            return
        m = (l + r) >> 1
        build(v << 1, l, m)
        build(v << 1 | 1, m, r)
        c1 = cnt[v << 1]
        c2 = cnt[v << 1 | 1]
        cnt[v] = c1 + c2
        sm[v] = sm[v << 1] + sm[v << 1 | 1]
        mn[v] = min(mn[v << 1], mn[v << 1 | 1])
        mx[v] = max(mx[v << 1], mx[v << 1 | 1])

    if n:
        build(1, 0, n)

    def apply(v, delta):
        c = cnt[v]
        if c:
            sm[v] += delta * c
            mn[v] += delta
            mx[v] += delta
            lazy[v] += delta

    def push(v):
        z = lazy[v]
        if z:
            apply(v << 1, z)
            apply(v << 1 | 1, z)
            lazy[v] = 0

    def pull(v):
        a = v << 1
        b = a | 1
        cnt[v] = cnt[a] + cnt[b]
        sm[v] = sm[a] + sm[b]
        mn[v] = min(mn[a], mn[b])
        mx[v] = max(mx[a], mx[b])

    def stats_rec(v, l, r, ql, qr):
        if qr <= l or r <= ql or cnt[v] == 0:
            return 0, 0, inf, -inf
        if ql <= l and r <= qr:
            return cnt[v], sm[v], mn[v], mx[v]
        push(v)
        m = (l + r) >> 1
        a = stats_rec(v << 1, l, m, ql, qr)
        b = stats_rec(v << 1 | 1, m, r, ql, qr)
        return (
            a[0] + b[0],
            a[1] + b[1],
            min(a[2], b[2]),
            max(a[3], b[3])
        )

    def range_stats(l, r):
        if l >= r or not n:
            return 0, 0, inf, -inf
        return stats_rec(1, 0, n, l, r)

    def add_rec(v, l, r, ql, qr, delta):
        if qr <= l or r <= ql or cnt[v] == 0:
            return
        if ql <= l and r <= qr:
            apply(v, delta)
            return
        push(v)
        m = (l + r) >> 1
        add_rec(v << 1, l, m, ql, qr, delta)
        add_rec(v << 1 | 1, m, r, ql, qr, delta)
        pull(v)

    def range_add(l, r, delta):
        if l < r and delta:
            add_rec(1, 0, n, l, r, delta)

    def extract_rec(v, l, r, ql, qr, out):
        if qr <= l or r <= ql or cnt[v] == 0:
            return
        if r - l == 1:
            out.append(l)
            return
        push(v)
        m = (l + r) >> 1
        extract_rec(v << 1, l, m, ql, qr, out)
        extract_rec(v << 1 | 1, m, r, ql, qr, out)
        pull(v)

    def extract(l, r):
        out = []
        if l < r and n:
            extract_rec(1, 0, n, l, r, out)
            for p in out:
                vals[p] = sm_point(p)
        return out

    def sm_point(p):
        v = 1
        l = 0
        r = n
        while r - l > 1:
            push(v)
            m = (l + r) >> 1
            if p < m:
                v <<= 1
                r = m
            else:
                v = v << 1 | 1
                l = m
        return sm[v]

    def remove_point(p):
        out = []
        extract_rec(1, 0, n, p, p + 1, out)
        vals[p] = sm_point(p)
        return vals[p]

    def materialize_rec(v, l, r):
        if cnt[v] == 0:
            return
        if r - l == 1:
            vals[l] = sm[v]
            return
        push(v)
        m = (l + r) >> 1
        materialize_rec(v << 1, l, m)
        materialize_rec(v << 1 | 1, m, r)

    treap_left = [-1] * n
    treap_right = [-1] * n
    treap_prio = [0] * n
    treap_key = list(range(n))
    roots = {}

    rnd = 88172645463325252
    mask64 = (1 << 64) - 1
    for i in range(n):
        rnd ^= (rnd << 7) & mask64
        rnd ^= rnd >> 9
        rnd ^= (rnd << 8) & mask64
        treap_prio[i] = rnd & mask64

    def treap_merge(a, b):
        if a == -1:
            return b
        if b == -1:
            return a
        if treap_prio[a] > treap_prio[b]:
            treap_right[a] = treap_merge(treap_right[a], b)
            return a
        treap_left[b] = treap_merge(a, treap_left[b])
        return b

    def treap_split(root, key):
        if root == -1:
            return -1, -1
        if treap_key[root] < key:
            a, b = treap_split(treap_right[root], key)
            treap_right[root] = a
            return root, b
        a, b = treap_split(treap_left[root], key)
        treap_left[root] = b
        return a, root

    def treap_insert(root, node):
        if root == -1:
            treap_left[node] = -1
            treap_right[node] = -1
            return node
        if treap_prio[node] > treap_prio[root]:
            a, b = treap_split(root, treap_key[node])
            treap_left[node] = a
            treap_right[node] = b
            return node
        if treap_key[node] < treap_key[root]:
            treap_left[root] = treap_insert(treap_left[root], node)
        else:
            treap_right[root] = treap_insert(treap_right[root], node)
        return root

    def treap_erase(root, key):
        if root == -1:
            return -1
        if key == treap_key[root]:
            return treap_merge(treap_left[root], treap_right[root])
        if key < treap_key[root]:
            treap_left[root] = treap_erase(treap_left[root], key)
        else:
            treap_right[root] = treap_erase(treap_right[root], key)
        return root

    def treap_prefix(root, key):
        res = 0
        while root != -1:
            if treap_key[root] <= key:
                res += 1
                x = treap_left[root]
                while x != -1:
                    res += 1
                    x = treap_right[x]
                root = treap_right[root]
            else:
                root = treap_left[root]
        return res

    def treap_range_count(root, l, r):
        if root == -1 or l >= r:
            return 0
        return treap_prefix(root, r - 1) - treap_prefix(root, l - 1)

    for p in range(n):
        key = (schema_of[p], tenants[p])
        roots[key] = treap_insert(roots.get(key, -1), p)

    read_results = []
    rollback_index = -1

    for ai, action in enumerate(actions):
        typ = action[0]

        if typ == "sum":
            _, schema, left_id, right_id = action
            si = schema_index[schema]
            if not (perm_masks[si] & 1):
                read_results.append(0)
                continue
            arr = schema_ids[si]
            l = bisect_left(arr, left_id)
            r = bisect_right(arr, right_id)
            if l >= r:
                read_results.append(0)
            else:
                gl = bases[si] + l
                gr = bases[si] + r
                read_results.append(range_stats(gl, gr)[1])

        elif typ == "get":
            _, schema, rid, mode = action
            si = schema_index[schema]
            if not (perm_masks[si] & 1):
                read_results.append(None)
                continue
            p = id_to_pos.get(rid, -1)
            if p == -1 or schema_of[p] != si or not live[p]:
                read_results.append(None)
                continue
            if mode == "S":
                if tenants[p] != subject_tenant:
                    read_results.append(None)
                    continue
            else:
                if not privileged:
                    read_results.append(None)
                    continue
            if tenants[p] == subject_tenant:
                read_results.append(sm_point(p))
            else:
                read_results.append(vals[p])

        elif typ == "add":
            _, schema, left_id, right_id, raw_delta = action
            si = schema_index[schema]
            if not (perm_masks[si] & 2):
                rollback_index = ai
                break
            arr = schema_ids[si]
            l = bisect_left(arr, left_id)
            r = bisect_right(arr, right_id)
            if l >= r:
                continue
            q = schema_params[si][2]
            if raw_delta >= 0:
                delta = (raw_delta // q) * q
            else:
                delta = -((-raw_delta) // q) * q
            if delta == 0:
                continue
            gl = bases[si] + l
            gr = bases[si] + r
            c, _, lo, hi = range_stats(gl, gr)
            if c == 0:
                continue
            low, high, _ = schema_params[si]
            if lo + delta < low or hi + delta > high:
                rollback_index = ai
                break
            range_add(gl, gr, delta)

        elif typ == "delete":
            _, schema, rid, mode = action
            si = schema_index[schema]
            p = id_to_pos.get(rid, -1)
            if p == -1 or schema_of[p] != si or not live[p]:
                rollback_index = ai
                break
            if mode == "S":
                if tenants[p] != subject_tenant:
                    rollback_index = ai
                    break
            else:
                if not privileged:
                    rollback_index = ai
                    break
            if not (perm_masks[si] & 4):
                rollback_index = ai
                break

            old_tenant = tenants[p]
            root_key = (si, old_tenant)
            roots[root_key] = treap_erase(roots.get(root_key, -1), p)
            if old_tenant == subject_tenant:
                remove_point(p)
            live[p] = False

        else:
            _, schema, left_id, right_id, target_tenant = action
            si = schema_index[schema]
            if not (perm_masks[si] & 8):
                rollback_index = ai
                break
            if target_tenant == subject_tenant:
                continue

            arr = schema_ids[si]
            l = bisect_left(arr, left_id)
            r = bisect_right(arr, right_id)
            if l >= r:
                continue

            gl = bases[si] + l
            gr = bases[si] + r
            target_root = roots.get((si, target_tenant), -1)
            if treap_range_count(target_root, gl, gr) > 0:
                rollback_index = ai
                break

            moved = extract(gl, gr)
            if not moved:
                continue

            source_key = (si, subject_tenant)
            source_root = roots.get(source_key, -1)
            target_key = (si, target_tenant)
            target_root = roots.get(target_key, -1)

            for p in moved:
                source_root = treap_erase(source_root, p)
                treap_left[p] = -1
                treap_right[p] = -1
                target_root = treap_insert(target_root, p)
                tenants[p] = target_tenant

            roots[source_key] = source_root
            roots[target_key] = target_root

    if rollback_index != -1:
        return rollback_index, read_results, initial_state

    if n:
        materialize_rec(1, 0, n)

    final_state = [None] * n
    for p in range(n):
        if live[p]:
            final_state[orig_at[p]] = [tenants[p], vals[p]]

    return -1, read_results, final_state