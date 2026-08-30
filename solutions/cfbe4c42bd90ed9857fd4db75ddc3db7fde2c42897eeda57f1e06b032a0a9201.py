from collections import deque


def audit_deck(var_count, cards, normal_edges, exception_edges, entry):
    n = len(cards)
    nsucc = [[] for _ in range(n)]
    esucc = [[] for _ in range(n)]
    for a, b in normal_edges:
        nsucc[a].append(b)
    for a, b in exception_edges:
        if cards[a][3]:
            esucc[a].append(b)

    reach = [False] * n
    reach[entry] = True
    dq = deque([entry])
    while dq:
        u = dq.popleft()
        for v in nsucc[u]:
            if not reach[v]:
                reach[v] = True
                dq.append(v)
        for v in esucc[u]:
            if not reach[v]:
                reach[v] = True
                dq.append(v)

    rd = [0] * n
    tg = [0] * n
    for i in range(n):
        if not reach[i]:
            continue
        m = 0
        for v in cards[i][0]:
            m |= 1 << v
        tg[i] = m
        m = 0
        for v in cards[i][1]:
            m |= 1 << v
        rd[i] = m

    nsr = [[] for _ in range(n)]
    esr = [[] for _ in range(n)]
    preds = [[] for _ in range(n)]
    for i in range(n):
        if not reach[i]:
            continue
        for v in nsucc[i]:
            if reach[v]:
                nsr[i].append(v)
                preds[v].append(i)
        for v in esucc[i]:
            if reach[v]:
                esr[i].append(v)
                preds[v].append(i)

    erased = [False] * n
    order = [i for i in range(n) if reach[i]]

    while True:
        live = [0] * n
        inq = [False] * n
        dq = deque()
        for i in reversed(order):
            dq.append(i)
            inq[i] = True
        while dq:
            u = dq.pop()
            inq[u] = False
            acc = 0
            for v in nsr[u]:
                acc |= live[v]
            acc &= ~tg[u]
            for v in esr[u]:
                acc |= live[v]
            acc |= rd[u]
            if acc != live[u]:
                live[u] = acc
                for p in preds[u]:
                    if not inq[p]:
                        inq[p] = True
                        dq.append(p)

        newly = []
        for i in order:
            if erased[i]:
                continue
            if tg[i] == 0:
                continue
            if cards[i][2] or cards[i][3]:
                continue
            acc = 0
            for v in nsr[i]:
                acc |= live[v]
            if tg[i] & acc == 0:
                newly.append(i)
        if not newly:
            break
        for i in newly:
            erased[i] = True
            tg[i] = 0
            rd[i] = 0

    out = []
    for i in range(n):
        if not reach[i]:
            out.append('U')
        elif erased[i]:
            out.append('D')
        else:
            out.append('K')
    return ''.join(out)