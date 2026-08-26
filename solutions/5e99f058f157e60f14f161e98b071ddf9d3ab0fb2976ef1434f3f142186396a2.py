def process_telemetry(initial, commands):
    import bisect

    names = {p[0] for p in initial}
    for c in commands:
        if c[0] == "add":
            names.add(c[1])
        elif c[0] == "rename":
            names.add(c[1])
            names.add(c[2])

    universe = sorted(names)
    pos = {name: i for i, name in enumerate(universe)}
    n = len(universe)

    size = 1
    while size < n:
        size <<= 1

    tree = [0] * (size << 1)
    lazy = bytearray(size << 1)

    for name, value in initial:
        tree[size + pos[name]] = value

    for i in range(size - 1, 0, -1):
        tree[i] = tree[i << 1] + tree[i << 1 | 1]

    def push(i):
        if lazy[i]:
            a = i << 1
            tree[a] = 0
            tree[a | 1] = 0
            lazy[a] = 1
            lazy[a | 1] = 1
            lazy[i] = 0

    def add(i, lo, hi, p, delta):
        if hi - lo == 1:
            tree[i] += delta
            lazy[i] = 0
            return
        push(i)
        mid = (lo + hi) >> 1
        if p < mid:
            add(i << 1, lo, mid, p, delta)
        else:
            add(i << 1 | 1, mid, hi, p, delta)
        tree[i] = tree[i << 1] + tree[i << 1 | 1]

    def total(i, lo, hi, ql, qr):
        if ql <= lo and hi <= qr:
            return tree[i]
        if qr <= lo or hi <= ql:
            return 0
        push(i)
        mid = (lo + hi) >> 1
        return total(i << 1, lo, mid, ql, qr) + total(i << 1 | 1, mid, hi, ql, qr)

    def clear(i, lo, hi, ql, qr):
        if ql <= lo and hi <= qr:
            tree[i] = 0
            lazy[i] = 1
            return
        if qr <= lo or hi <= ql:
            return
        push(i)
        mid = (lo + hi) >> 1
        clear(i << 1, lo, mid, ql, qr)
        clear(i << 1 | 1, mid, hi, ql, qr)
        tree[i] = tree[i << 1] + tree[i << 1 | 1]

    current = {name: [pos[name], value] for name, value in initial}
    answer = []

    for c in commands:
        typ = c[0]

        if typ == "add":
            name = c[1]
            delta = c[2]
            item = current[name]
            item[1] += delta
            add(1, 0, size, item[0], delta)

        elif typ == "rename":
            old = c[1]
            new = c[2]
            item = current.pop(old)
            p_old, value = item
            p_new = pos[new]
            if value:
                add(1, 0, size, p_old, -value)
                add(1, 0, size, p_new, value)
            item[0] = p_new
            current[new] = item

        else:
            legacy = c[1]
            selectors = c[2]
            reset = c[3]

            if selectors:
                effective = [s for s in selectors if s]
            elif legacy:
                effective = [legacy]
            else:
                effective = []

            if not effective:
                answer.append(0)
                continue

            intervals = []
            for s in effective:
                left = bisect.bisect_left(universe, s)
                right = bisect.bisect_left(universe, s + "\U0010ffff")
                if left < right:
                    intervals.append((left, right))

            if not intervals:
                answer.append(0)
                continue

            intervals.sort()
            merged = []
            left, right = intervals[0]

            for a, b in intervals[1:]:
                if a <= right:
                    if b > right:
                        right = b
                else:
                    merged.append((left, right))
                    left, right = a, b
            merged.append((left, right))

            result = 0
            for left, right in merged:
                result += total(1, 0, size, left, right)

            answer.append(result)

            if reset:
                for left, right in merged:
                    clear(1, 0, size, left, right)

    return answer