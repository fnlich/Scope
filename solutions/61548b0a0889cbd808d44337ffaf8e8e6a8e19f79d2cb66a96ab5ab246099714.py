def simulate_cells(cell_count, programs, initial_jobs):
    from collections import deque

    cells = [None] * cell_count
    waiters = [None] * cell_count
    emitted = []
    rejected = []
    queue = deque()

    jp = []
    pc = []
    env = []
    stacks = []
    status = []
    wait_token = []
    blocked_cells = []

    for p, e in initial_jobs:
        jid = len(jp)
        jp.append(p)
        pc.append(0)
        env.append(e)
        stacks.append([])
        status.append(0)
        wait_token.append(0)
        blocked_cells.append(None)
        queue.append(jid)

    token = 0
    mask = (1 << 20) - 1

    while queue:
        jid = queue.popleft()
        if status[jid] != 0:
            continue

        while True:
            pos = pc[jid]
            prog = programs[jp[jid]]
            if pos >= len(prog):
                status[jid] = 2
                break

            ins = prog[pos]
            pc[jid] = pos + 1
            op = ins[0]
            e = env[jid]

            if op == "push":
                stacks[jid].append(e)
                env[jid] = ins[1] * e + ins[2]

            elif op == "pop":
                env[jid] = stacks[jid].pop()

            elif op == "wait":
                c, a, b = ins[1], ins[2], ins[3]
                x = cells[c]
                if x is not None:
                    env[jid] = a * e + b * x
                else:
                    token += 1
                    wait_token[jid] = token
                    blocked_cells[jid] = (c,)
                    status[jid] = 1
                    w = waiters[c]
                    entry = (token << 20) | jid
                    if w is None:
                        waiters[c] = [entry]
                    else:
                        w.append(entry)
                    break

            elif op == "wait_any":
                cs, a, b, d = ins[1], ins[2], ins[3], ins[4]
                chosen = None
                x = None
                for c in cs:
                    v = cells[c]
                    if v is not None:
                        chosen = c
                        x = v
                        break
                if chosen is not None:
                    env[jid] = a * e + b * x + d * chosen
                else:
                    token += 1
                    wait_token[jid] = token
                    bc = tuple(cs)
                    blocked_cells[jid] = bc
                    status[jid] = 1
                    entry = (token << 20) | jid
                    for c in bc:
                        w = waiters[c]
                        if w is None:
                            waiters[c] = [entry]
                        else:
                            w.append(entry)
                    break

            elif op == "fill":
                c, a, b = ins[1], ins[2], ins[3]
                x = a * e + b
                if cells[c] is not None:
                    rejected.append((jid, c))
                else:
                    cells[c] = x
                    w = waiters[c]
                    waiters[c] = None
                    if w is not None:
                        for entry in w:
                            other = entry & mask
                            t = entry >> 20
                            if status[other] == 1 and wait_token[other] == t:
                                status[other] = 0
                                queue.append(other)

            elif op == "fork":
                child = len(jp)
                jp.append(ins[1])
                pc.append(0)
                env.append(e)
                stacks.append(stacks[jid].copy())
                status.append(0)
                wait_token.append(0)
                blocked_cells.append(None)
                queue.append(child)

            elif op == "emit":
                emitted.append((jid, e))

    blocked = []
    for jid in range(len(jp)):
        if status[jid] == 1:
            blocked.append((jid, blocked_cells[jid]))

    return emitted, rejected, cells, blocked