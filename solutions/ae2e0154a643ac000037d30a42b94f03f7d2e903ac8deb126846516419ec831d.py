def reconcile_poll_document(document: str) -> str:
    import heapq

    lines = document.splitlines()
    n = len(lines)
    size = 1
    while size < max(1, n):
        size <<= 1
    inf = 10**30
    clock_tree = [inf] * (2 * size)

    desc = {}
    subs = []
    active = []
    sid_to_idx = {}
    read_heaps = {}
    write_heaps = {}
    read_global = []
    write_global = []
    output = []

    def tree_update(pos, value):
        p = pos + size
        clock_tree[p] = value
        p >>= 1
        while p:
            left = p << 1
            v = clock_tree[left]
            w = clock_tree[left | 1]
            clock_tree[p] = v if v < w else w
            p >>= 1

    def first_clock(t):
        if clock_tree[1] > t:
            return -1
        p = 1
        while p < size:
            left = p << 1
            if clock_tree[left] <= t:
                p = left
            else:
                p = left | 1
        return p - size

    def clean_read(fd):
        h = read_heaps.get(fd)
        if not h:
            return -1
        while h and not active[h[0]]:
            heapq.heappop(h)
        return h[0] if h else -1

    def clean_write(fd):
        h = write_heaps.get(fd)
        if not h:
            return -1
        while h and not active[h[0]]:
            heapq.heappop(h)
        return h[0] if h else -1

    def ready_read(fd):
        d = desc[fd]
        return d[0] == 0 or d[1] > 0 or d[3] == 1

    def ready_write(fd):
        d = desc[fd]
        return d[0] == 0 or d[2] == 1 or d[3] == 1

    def refresh_read(fd):
        idx = clean_read(fd)
        if idx >= 0 and ready_read(fd):
            heapq.heappush(read_global, (idx, fd))

    def refresh_write(fd):
        idx = clean_write(fd)
        if idx >= 0 and ready_write(fd):
            heapq.heappush(write_global, (idx, fd))

    def read_candidate():
        while read_global:
            idx, fd = read_global[0]
            cur = clean_read(fd)
            if cur != idx or not active[idx] or not ready_read(fd):
                heapq.heappop(read_global)
                if cur >= 0 and active[cur] and ready_read(fd):
                    heapq.heappush(read_global, (cur, fd))
                continue
            return idx, fd
        return -1, None

    def write_candidate():
        while write_global:
            idx, fd = write_global[0]
            cur = clean_write(fd)
            if cur != idx or not active[idx] or not ready_write(fd):
                heapq.heappop(write_global)
                if cur >= 0 and active[cur] and ready_write(fd):
                    heapq.heappush(write_global, (cur, fd))
                continue
            return idx, fd
        return -1, None

    for line in lines:
        if not line.strip():
            continue
        a = line.split()
        op = a[0]

        if op == "DESC":
            fd = int(a[2])
            kind = 0 if a[3] == "REG" else 1
            desc[fd] = [kind, int(a[4]), int(a[5]), int(a[6])]

        elif op == "SET":
            fd = int(a[2])
            d = desc[fd]
            d[1] = int(a[3])
            d[2] = int(a[4])
            d[3] = int(a[5])
            refresh_read(fd)
            refresh_write(fd)

        elif op == "ADD":
            t = int(a[1])
            sid = int(a[2])
            token = int(a[3])
            typ = a[4]
            idx = len(subs)

            if typ == "CLOCK":
                value = int(a[6])
                precision = int(a[7])
                deadline = value if a[5] == "ABS" else t + value
                threshold = deadline - precision
                rec = [typ, sid, token, None, 0, deadline, threshold]
                subs.append(rec)
                active.append(True)
                sid_to_idx[sid] = idx
                tree_update(idx, threshold)

            elif typ == "READ":
                fd = int(a[5])
                capacity = int(a[6])
                rec = [typ, sid, token, fd, capacity, 0, 0]
                subs.append(rec)
                active.append(True)
                sid_to_idx[sid] = idx
                heapq.heappush(read_heaps.setdefault(fd, []), idx)
                refresh_read(fd)

            else:
                fd = int(a[5])
                rec = [typ, sid, token, fd, 0, 0, 0]
                subs.append(rec)
                active.append(True)
                sid_to_idx[sid] = idx
                heapq.heappush(write_heaps.setdefault(fd, []), idx)
                refresh_write(fd)

        elif op == "DROP":
            sid = int(a[2])
            idx = sid_to_idx.get(sid)
            if idx is not None and active[idx]:
                active[idx] = False
                if subs[idx][0] == "CLOCK":
                    tree_update(idx, inf)

        else:
            t = int(a[1])
            limit = int(a[2])
            for _ in range(limit):
                ri, rfd = read_candidate()
                wi, wfd = write_candidate()
                ci = first_clock(t)

                best = -1
                kind = None
                fd = None

                if ri >= 0:
                    best, kind, fd = ri, "READ", rfd
                if wi >= 0 and (best < 0 or wi < best):
                    best, kind, fd = wi, "WRITE", wfd
                if ci >= 0 and (best < 0 or ci < best):
                    best, kind, fd = ci, "CLOCK", None

                if best < 0:
                    break

                rec = subs[best]
                token = rec[2]

                if kind == "CLOCK":
                    output.append(f"{t} {rec[1]} {token} CLOCK {rec[5]}")
                    active[best] = False
                    tree_update(best, inf)

                elif kind == "READ":
                    h = read_heaps[fd]
                    while h and not active[h[0]]:
                        heapq.heappop(h)
                    if not h or h[0] != best:
                        refresh_read(fd)
                        continue

                    heapq.heappop(h)
                    d = desc[fd]
                    available = d[1]
                    capacity = rec[4]
                    bytes_read = min(available, capacity)
                    flags = (1 if d[3] else 0) + (2 if available > capacity else 0) + (4 if d[0] == 0 else 0)
                    output.append(f"{t} {rec[1]} {token} READ {bytes_read} {flags}")
                    d[1] -= bytes_read
                    active[best] = False
                    refresh_read(fd)

                else:
                    h = write_heaps[fd]
                    while h and not active[h[0]]:
                        heapq.heappop(h)
                    if not h or h[0] != best:
                        refresh_write(fd)
                        continue

                    heapq.heappop(h)
                    d = desc[fd]
                    flags = (1 if d[3] else 0) + (4 if d[0] == 0 else 0)
                    output.append(f"{t} {rec[1]} {token} WRITE {flags}")
                    active[best] = False
                    refresh_write(fd)

    return "\n".join(output)