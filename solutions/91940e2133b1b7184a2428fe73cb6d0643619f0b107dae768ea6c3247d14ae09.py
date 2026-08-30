def track_fragments(starts, operations, modulus, limits):
    half = modulus // 2
    messages = []
    gaps = []
    reports = []
    discarded = []
    evicted = []

    nxt = [starts[0], starts[1]]
    info = [{}, {}]
    active = [{}, {}]
    used = [0, 0]
    partial = [{}, {}]

    def discard_partials(d):
        for s in sorted(partial[d].keys()):
            p = partial[d][s]
            discarded.append((d, s, p[0] % modulus, p[1] % modulus, "gap"))
        partial[d].clear()

    def consume(d, pos):
        stream, begin, end, payload = info[d][pos][0], info[d][pos][1], info[d][pos][2], info[d][pos][3]
        if begin:
            if stream in partial[d]:
                p = partial[d][stream]
                discarded.append((d, stream, p[0] % modulus, p[1] % modulus, "restart"))
                del partial[d][stream]
            partial[d][stream] = [pos, pos, [payload]]
        else:
            if stream not in partial[d]:
                discarded.append((d, stream, pos % modulus, pos % modulus, "orphan"))
                return
            p = partial[d][stream]
            p[1] = pos
            p[2].append(payload)
        if end:
            p = partial[d][stream]
            messages.append((d, stream, p[0] % modulus, p[1] % modulus, b"".join(p[2])))
            del partial[d][stream]

    def drain(d):
        while nxt[d] in active[d]:
            pos = nxt[d]
            del active[d][pos]
            used[d] -= len(info[d][pos][3])
            nxt[d] = pos + 1
            consume(d, pos)

    def enforce(d, opidx):
        while used[d] > limits[d] and active[d]:
            pos = max(active[d].keys())
            del active[d][pos]
            ln = len(info[d][pos][3])
            used[d] -= ln
            evicted.append((opidx, d, pos % modulus, ln))

    for i, op in enumerate(operations):
        if op[0] == "D":
            _, d, tsn, stream, begin, end, payload = op
            cur = nxt[d]
            delta = (tsn - cur) % modulus
            if delta > half:
                delta -= modulus
            pos = cur + delta
            if pos in info[d]:
                rec = info[d][pos]
                same = (rec[0] == stream and bool(rec[1]) == bool(begin) and bool(rec[2]) == bool(end) and rec[3] == payload)
                if same:
                    reports.append((i, d, tsn % modulus, "same"))
                    if pos >= nxt[d] and pos not in active[d]:
                        active[d][pos] = True
                        used[d] += len(payload)
                        drain(d)
                        enforce(d, i)
                else:
                    reports.append((i, d, tsn % modulus, "conflict"))
            else:
                info[d][pos] = (stream, bool(begin), bool(end), payload)
                if pos < nxt[d]:
                    reports.append((i, d, tsn % modulus, "late"))
                else:
                    active[d][pos] = True
                    used[d] += len(payload)
                    drain(d)
                    enforce(d, i)
        else:
            _, d, target_tsn = op
            cur = nxt[d]
            delta = (target_tsn - cur) % modulus
            if delta == 0:
                delta = modulus
            target = cur + delta
            pos = cur
            run_start = None
            while pos < target:
                if pos in active[d]:
                    if run_start is not None:
                        gaps.append((d, run_start % modulus, pos % modulus, pos - run_start))
                        discard_partials(d)
                        run_start = None
                    del active[d][pos]
                    used[d] -= len(info[d][pos][3])
                    consume(d, pos)
                else:
                    if run_start is None:
                        run_start = pos
                pos += 1
            if run_start is not None:
                gaps.append((d, run_start % modulus, target % modulus, target - run_start))
                discard_partials(d)
            nxt[d] = target
            drain(d)

    return {"messages": messages, "gaps": gaps, "reports": reports, "discarded": discarded, "evicted": evicted}