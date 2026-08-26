def merge_replicated_topk(shard_offsets, shard_sizes, shard_weights, queries, ks, mode, min_support, min_separations):
    from bisect import bisect_left, bisect_right, insort

    s_count = len(shard_offsets)
    largest = mode == "largest"
    results = []

    for q, shard_lists in enumerate(queries):
        groups = {}

        for s in range(s_count):
            offset = shard_offsets[s]
            weight = shard_weights[s]
            for value, local_index, generation in shard_lists[s]:
                gidx = offset + local_index
                entry = groups.get(gidx)
                if entry is None:
                    groups[gidx] = [generation, [(value, s, weight)]]
                elif generation > entry[0]:
                    entry[0] = generation
                    entry[1] = [(value, s, weight)]
                elif generation == entry[0]:
                    entry[1].append((value, s, weight))

        eligible = []
        for gidx, entry in groups.items():
            reports = entry[1]
            if len(reports) < min_support:
                continue

            reports.sort(key=lambda x: (x[0], x[1]))
            total_weight = 0
            for _, _, w in reports:
                total_weight += w
            target = (total_weight + 1) // 2

            cumulative = 0
            median_value = reports[-1][0]
            for value, _, w in reports:
                cumulative += w
                if cumulative >= target:
                    median_value = value
                    break

            eligible.append((median_value, gidx))

        if largest:
            eligible.sort(key=lambda x: (-x[0], x[1]))
        else:
            eligible.sort(key=lambda x: (x[0], x[1]))

        separation = min_separations[q]
        need = ks[q]
        admitted_positions = []
        answer = []

        for value, gidx in eligible:
            pos = bisect_left(admitted_positions, gidx)
            if pos > 0 and gidx - admitted_positions[pos - 1] < separation:
                continue
            if pos < len(admitted_positions) and admitted_positions[pos] - gidx < separation:
                continue

            admitted_positions.insert(pos, gidx)
            answer.append((value, gidx))
            if len(answer) == need:
                break

        results.append(answer)

    return results