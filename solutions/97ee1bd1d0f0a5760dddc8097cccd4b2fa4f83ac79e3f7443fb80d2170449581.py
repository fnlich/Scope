def plan_workflow(authenticated, states, transitions, field_features, project_features, entitlements, user_roles, start, target, initial_fields, values):
    if not authenticated:
        return {"status": "unauthenticated"}

    if start == target:
        return {
            "status": "ok",
            "transitions": [],
            "fields": [[k, initial_fields[k]] for k in sorted(initial_fields)]
        }

    state_index = {s[0]: i for i, s in enumerate(states)}
    ns = len(states)

    fields = set(initial_fields)
    fields.update(values)
    fields.update(field_features)
    for t in transitions:
        fields.update(t["present"])
        fields.update(t["absent"])
        fields.update(t["prompt"])
        fields.update(t["remove"])
        fields.update(t["equals"])
    fields = sorted(fields)
    nf = len(fields)
    fi = {f: i for i, f in enumerate(fields)}

    domains = []
    value_maps = []
    for f in fields:
        vals = []
        seen_vals = set()
        if f in initial_fields:
            v = initial_fields[f]
            k = (type(v), v)
            if k not in seen_vals:
                seen_vals.add(k)
                vals.append(v)
        if f in values:
            v = values[f]
            k = (type(v), v)
            if k not in seen_vals:
                seen_vals.add(k)
                vals.append(v)
        domains.append([None] + vals)
        vm = {}
        for j, v in enumerate(vals, 1):
            vm[(type(v), v)] = j
        value_maps.append(vm)

    shifts = [0] * nf
    masks = [0] * nf
    total_bits = 0
    for i, d in enumerate(domains):
        r = len(d)
        w = (r - 1).bit_length()
        shifts[i] = total_bits
        masks[i] = (1 << w) - 1 if w else 0
        total_bits += w

    base = 1 << total_bits
    fullmask = base - 1

    initial_code = 0
    for i, f in enumerate(fields):
        if f in initial_fields:
            v = initial_fields[f]
            c = value_maps[i][(type(v), v)]
            initial_code |= c << shifts[i]

    ent = set(entitlements)
    project_feature_sets = {p: set(v) for p, v in project_features.items()}
    role_sets = {p: set(v) for p, v in user_roles.items()}

    raw_outs = [[] for _ in range(ns)]

    for t in transitions:
        sidx = state_index[t["from"]]
        tidx = state_index[t["to"]]
        src_project = states[sidx][1]
        dst_project = states[tidx][1]

        if not all(r in role_sets.get(src_project, set()) for r in t["source_roles"]):
            continue
        if not all(r in role_sets.get(dst_project, set()) for r in t["target_roles"]):
            continue

        dst_features = project_feature_sets[dst_project]
        bad = False

        for f in t["prompt"]:
            if f not in values:
                bad = True
                break
            feat = field_features.get(f)
            if feat is not None and (feat not in ent or feat not in dst_features):
                bad = True
                break
        if bad:
            continue

        cons = {}
        for f in t["absent"]:
            cons[fi[f]] = (0, 0)
        for f in t["present"]:
            if fi[f] not in cons:
                cons[fi[f]] = (1, 0)

        for f, v in t["equals"].items():
            idx = fi[f]
            c = value_maps[idx].get((type(v), v))
            if c is None:
                bad = True
                break
            cons[idx] = (2, c)

        if bad:
            continue

        clear = 0
        setbits = 0

        for f in t["prompt"]:
            idx = fi[f]
            c = value_maps[idx][(type(values[f]), values[f])]
            if masks[idx]:
                clear |= masks[idx] << shifts[idx]
                setbits |= c << shifts[idx]

        for f in t["remove"]:
            idx = fi[f]
            if masks[idx]:
                clear |= masks[idx] << shifts[idx]

        keep = fullmask ^ clear
        raw_outs[sidx].append((t["id"], tidx, cons, keep, setbits))

    outs = [[] for _ in range(ns)]
    checks = [[] for _ in range(ns)]

    for s in range(ns):
        arr = raw_outs[s]
        arr.sort(key=lambda x: x[0])
        outs[s] = [(x[0], x[1], x[3], x[4]) for x in arr]

        m = len(arr)
        if not m:
            continue

        allbits = (1 << m) - 1
        unconstrained = [allbits] * nf
        absent_bits = [0] * nf
        present_bits = [0] * nf
        equal_bits = [{} for _ in range(nf)]
        constrained = [False] * nf

        for j, x in enumerate(arr):
            bit = 1 << j
            for idx, (kind, code) in x[2].items():
                constrained[idx] = True
                unconstrained[idx] &= ~bit
                if kind == 0:
                    absent_bits[idx] |= bit
                elif kind == 1:
                    present_bits[idx] |= bit
                else:
                    d = equal_bits[idx]
                    d[code] = d.get(code, 0) | bit

        checks[s] = [
            (i, unconstrained[i], absent_bits[i], present_bits[i], equal_bits[i])
            for i in range(nf) if constrained[i]
        ]

    start_key = (state_index[start] << total_bits) | initial_code
    target_index = state_index[target]

    keys = [start_key]
    parents = [-1]
    parent_trans = [-1]
    seen = {start_key: 0}

    head = 0
    while head < len(keys):
        node = head
        key = keys[node]
        state = key >> total_bits
        cfg = key & fullmask
        head += 1

        cand = (1 << len(outs[state])) - 1
        for idx, ub, ab, pb, eb in checks[state]:
            code = (cfg >> shifts[idx]) & masks[idx]
            if code == 0:
                allowed = ub | ab
            else:
                allowed = ub | pb | eb.get(code, 0)
            cand &= allowed
            if not cand:
                break