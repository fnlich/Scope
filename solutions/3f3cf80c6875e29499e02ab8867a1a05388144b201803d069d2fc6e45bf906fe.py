def canonical_spinner_cycles(animations):
    from bisect import bisect_right

    def min_rotation(a):
        m = len(a)
        if m <= 1:
            return 0
        i, j, k = 0, 1, 0
        while i < m and j < m and k < m:
            x = a[(i + k) % m]
            y = a[(j + k) % m]
            if x == y:
                k += 1
            elif x < y:
                j = j + k + 1
                if i == j:
                    j += 1
                k = 0
            else:
                i = i + k + 1
                if i == j:
                    i += 1
                k = 0
        return min(i, j)

    result = []

    for animation in animations:
        frames = animation["frames"]
        intervals = animation["intervals"]
        elapsed = animation["elapsed"]
        n = len(frames)

        if n == 0:
            result.append({
                "cycle": [],
                "intervals": [],
                "copies": 0,
                "active": []
            })
            continue

        records = list(zip(frames, intervals))

        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j and records[i] != records[j]:
                j = pi[j - 1]
            if records[i] == records[j]:
                j += 1
            pi[i] = j

        p = n - pi[-1]
        if n % p:
            p = n

        copies = n // p
        base_frames = frames[:p]
        base_intervals = intervals[:p]
        forward = list(zip(base_frames, base_intervals))

        fi = min_rotation(forward)
        forward_canonical = [
            forward[(fi + j) % p]
            for j in range(p)
        ]

        reverse_edge_records = [
            (base_frames[i], base_intervals[(i - 1) % p])
            for i in range(p)
        ]
        reverse_sequence = [
            reverse_edge_records[(-j) % p]
            for j in range(p)
        ]

        ri = min_rotation(reverse_sequence)
        reverse_canonical = [
            reverse_sequence[(ri + j) % p]
            for j in range(p)
        ]

        canonical = (
            forward_canonical
            if forward_canonical <= reverse_canonical
            else reverse_canonical
        )

        cycle = [record[0] for record in canonical]
        canonical_intervals = [record[1] for record in canonical]

        total = 0
        boundaries = []
        for interval in canonical_intervals:
            total += interval
            boundaries.append(total)

        active = []
        for t in elapsed:
            x = t % total
            active.append(cycle[bisect_right(boundaries, x)])

        result.append({
            "cycle": cycle,
            "intervals": canonical_intervals,
            "copies": copies,
            "active": active
        })

    return result