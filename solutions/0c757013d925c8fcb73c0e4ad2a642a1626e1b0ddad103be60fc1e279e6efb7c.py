def build_extension_dispatch(records, start_costs, transition_costs):
    n = len(records)
    if n == 0:
        return (0, None)

    L = len(start_costs)
    exts = [r[0] for r in records]
    media = [r[1] for r in records]

    vals = [[None] * n for _ in range(L)]
    for p in range(L):
        k = p + 1
        row = vals[p]
        for i, s in enumerate(exts):
            row[i] = s[-k] if k <= len(s) else ""

    size = 1 << n
    inf = 10**100

    dp = [[0] * L for _ in range(size)]
    choice_p = [[-1] * L for _ in range(size)]
    choice_s = [[None] * L for _ in range(size)]

    masks_by_count = [[] for _ in range(n + 1)]
    for m in range(1, size):
        masks_by_count[m.bit_count()].append(m)

    for mask in masks_by_count[1]:
        i = (mask & -mask).bit_length() - 1
        for q in range(L):
            dp[mask][q] = 0

    for count in range(2, n + 1):
        for mask in masks_by_count[count]:
            best_cont = [inf] * L
            best_sym = [None] * L

            for p in range(L):
                groups = {}
                bits = mask
                vr = vals[p]
                while bits:
                    b = bits & -bits
                    i = b.bit_length() - 1
                    s = vr[i]
                    groups[s] = groups.get(s, 0) | b
                    bits ^= b

                bc = inf
                bs = None
                for s, yes in groups.items():
                    if yes == mask:
                        continue
                    no = mask ^ yes
                    v = dp[yes][p]
                    w = dp[no][p]
                    cont = v if v >= w else w
                    if cont < bc:
                        bc = cont
                        bs = s
                    elif cont == bc:
                        if bs is None or (s == "" and bs != "") or (s != "" and bs != "" and s < bs):
                            bs = s

                best_cont[p] = bc
                best_sym[p] = bs

            row = dp[mask]
            for q in range(L):
                bv = inf
                bp = -1
                for p in range(L):
                    c = best_cont[p]
                    if c == inf:
                        continue
                    v = transition_costs[q][p] + c
                    if v < bv or (v == bv and (bp == -1 or p < bp)):
                        bv = v
                        bp = p
                row[q] = bv
                choice_p[mask][q] = bp
                choice_s[mask][q] = best_sym[bp] if bp >= 0 else None

    full = size - 1
    root_cost = inf
    root_p = -1
    root_s = None
    for p in range(L):
        if best_cont := None:
            pass

    root_best_cont = [inf] * L
    root_best_sym = [None] * L
    for p in range(L):
        groups = {}
        bits = full
        vr = vals[p]
        while bits:
            b = bits & -bits
            i = b.bit_length() - 1
            s = vr[i]
            groups[s] = groups.get(s, 0) | b
            bits ^= b

        bc = inf
        bs = None
        for s, yes in groups.items():
            if yes == full:
                continue
            no = full ^ yes
            v = dp[yes][p]
            w = dp[no][p]
            cont = v if v >= w else w
            if cont < bc:
                bc = cont
                bs = s
            elif cont == bc:
                if bs is None or (s == "" and bs != "") or (s != "" and bs != "" and s < bs):
                    bs = s
        root_best_cont[p] = bc
        root_best_sym[p] = bs

    for p in range(L):
        if root_best_cont[p] == inf:
            continue
        v = start_costs[p] + root_best_cont[p]
        if v < root_cost or (v == root_cost and (root_p == -1 or p < root_p)):
            root_cost = v
            root_p = p
            root_s = root_best_sym[p]

    def make_tree(mask, prev):
        if mask & (mask - 1) == 0:
            i = (mask & -mask).bit_length() - 1
            return ("result", exts[i], media[i])
        p = choice_p[mask][prev]
        s = choice_s[mask][prev]
        yes = 0
        bits = mask
        vr = vals[p]
        while bits:
            b = bits & -bits
            i = b.bit_length() - 1
            if vr[i] == s:
                yes |= b
            bits ^= b
        no = mask ^ yes
        return ("ask", p, s, make_tree(yes, p), make_tree(no, p))

    tree = make_tree(full, root_p)
    return (root_cost, tree)