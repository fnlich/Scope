def adapt_publications(n, initial_new, initial_legacy, events):
    NEW_revision = [0]*n
    legacy_present = bytearray(n)

    size = 1
    while size < n:
        size *= 2
    if size == 0:
        size = 1

    leaf_attr = [0]*size
    leaf_active = bytearray(size)
    active_count = [0]*(2*size)
    lazy_clear = [0]*(2*size)
    lazy_set = [0]*(2*size)
    has_lazy = bytearray(2*size)

    def apply_lazy(node, clear_mask, set_mask):
        if node >= size:
            idx = node - size
            if leaf_active[idx]:
                a = leaf_attr[idx]
                a = ((a & ~clear_mask) | set_mask) & 0xffffffff
                leaf_attr[idx] = a
            return
        if has_lazy[node]:
            oc = lazy_clear[node]
            os_ = lazy_set[node]
            nc = oc | clear_mask
            ns = (os_ & ~clear_mask) | set_mask
            lazy_clear[node] = nc
            lazy_set[node] = ns & 0xffffffff
        else:
            lazy_clear[node] = clear_mask
            lazy_set[node] = set_mask & 0xffffffff
            has_lazy[node] = 1

    def push_down(node):
        if has_lazy[node]:
            c = lazy_clear[node]
            s = lazy_set[node]
            apply_lazy(2*node, c, s)
            apply_lazy(2*node+1, c, s)
            has_lazy[node] = 0
            lazy_clear[node] = 0
            lazy_set[node] = 0

    def pull_up(node):
        active_count[node] = active_count[2*node] + active_count[2*node+1]

    def activate_leaf(pos, attribute):
        node = 1
        lo, hi = 0, size - 1
        path = []
        while node < size:
            push_down(node)
            path.append(node)
            mid = (lo + hi)//2
            if pos <= mid:
                node = 2*node
                hi = mid
            else:
                node = 2*node+1
                lo = mid+1
        idx = node - size
        leaf_active[idx] = 1
        leaf_attr[idx] = attribute & 0xffffffff
        active_count[node] = 1
        for p in reversed(path):
            pull_up(p)

    def range_apply(l, r, clear_mask, set_mask):
        def rec(node, lo, hi):
            if r < lo or hi < l:
                return
            if l <= lo and hi <= r:
                if active_count[node] == 0:
                    return
                if active_count[node] == (hi - lo + 1):
                    apply_lazy(node, clear_mask, set_mask)
                    return
            push_down(node)
            mid = (lo+hi)//2
            rec(2*node, lo, mid)
            rec(2*node+1, mid+1, hi)
            pull_up(node)
        rec(1, 0, size-1)

    def get_leaf_attr(pos):
        node = 1
        lo, hi = 0, size - 1
        while node < size:
            push_down(node)
            mid = (lo+hi)//2
            if pos <= mid:
                node = 2*node
                hi = mid
            else:
                node = 2*node+1
                lo = mid+1
        idx = node - size
        return leaf_attr[idx]

    parent = list(range(n+1))
    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    parent_leg = list(range(n+1))
    def find_leg(x):
        root = x
        while parent_leg[root] != root:
            root = parent_leg[root]
        while parent_leg[x] != root:
            parent_leg[x], x = root, parent_leg[x]
        return root

    tickets = []
    ticket_head = 0

    new_list = sorted(initial_new, key=lambda t: t[0])
    for h, rev, attrib in new_list:
        NEW_revision[h] = rev
        activate_leaf(h, attrib)
        parent[h] = h+1
        tickets.append((h, rev))

    for h in initial_legacy:
        legacy_present[h] = 1
        parent_leg[h] = h+1

    template_tag = 0
    serial = 0
    results = []

    for ev in events:
        typ = ev[0]
        if typ == "P":
            _, l, r, revision, attributes = ev
            idx = find(l)
            while idx <= r:
                NEW_revision[idx] = revision
                activate_leaf(idx, attributes)
                tickets.append((idx, revision))
                parent[idx] = idx+1
                idx = find(idx+1)
        elif typ == "A":
            _, l, r, clear_mask, set_mask = ev
            range_apply(l, r, clear_mask, set_mask)
        elif typ == "L":
            _, l, r = ev
            idx = find_leg(l)
            while idx <= r:
                legacy_present[idx] = 1
                parent_leg[idx] = idx+1
                idx = find_leg(idx+1)
        elif typ == "T":
            _, tag = ev
            template_tag = tag
        elif typ == "D":
            _, k = ev
            remaining = len(tickets) - ticket_head
            take = k if k < remaining else remaining
            end = ticket_head + take
            for i in range(ticket_head, end):
                h, rev = tickets[i]
                if legacy_present[h]:
                    continue
                x = get_leaf_attr(h)
                shared = x & 0x0000ffff
                a = (x >> 16) & 0x1f
                legacy_attr = shared | (1 << (16 + a))
                serial += 1
                results.append((serial, h, rev, template_tag, legacy_attr))
            ticket_head = end

    return results