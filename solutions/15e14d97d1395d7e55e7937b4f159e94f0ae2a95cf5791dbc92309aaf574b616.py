def run_session_trace(events):
    P = 1000000007
    keys = set()
    for e in events:
        t = e[0]
        if t == "request":
            keys.add(e[2])
        elif t == "delegate":
            keys.add(e[2])
            keys.add(e[4])
    w = {}
    i = 1
    for k in sorted(keys):
        w[k] = i
        i += 1
    var = [{}]
    sig = [0]
    lreq = [None]
    lres = [None]
    acct = [0]
    tot = [0]
    pas = [0]
    serial = 0
    for e in events:
        t = e[0]
        if t == "new":
            var.append({})
            sig.append(0)
            lreq.append(None)
            lres.append(None)
            acct.append(len(tot))
            tot.append(0)
            pas.append(0)
        elif t == "dup":
            s = e[1]
            var.append(dict(var[s]))
            sig.append(sig[s])
            lreq.append(lreq[s])
            lres.append(lres[s])
            acct.append(acct[s])
        elif t == "reset":
            s = e[1]
            var[s] = {}
            sig[s] = 0
            lreq[s] = None
            lres[s] = None
        elif t == "request":
            serial += 1
            s = e[1]
            k = e[2]
            d = e[3]
            old = var[s].get(k, 0)
            nv = old + d
            var[s][k] = nv
            sig[s] = (sig[s] + w[k] * d) % P
            st = 200 if nv >= 0 else 409
            lres[s] = [st, nv, sig[s]]
            lreq[s] = [serial, "request", k, d]
        elif t == "delegate":
            serial += 1
            s = e[1]
            k = e[2]
            src = e[3]
            sk = e[4]
            off = e[5]
            base = var[src].get(sk, 0)
            nv = base + off
            old = var[s].get(k, 0)
            var[s][k] = nv
            sig[s] = (sig[s] + w[k] * (nv - old)) % P
            st = 200 if nv >= 0 else 409
            lres[s] = [st, nv, sig[s]]
            lreq[s] = [serial, "delegate", k, src, sk, off]
        elif t == "assert":
            s = e[1]
            a = acct[s]
            tot[a] += 1
            r = lres[s]
            if r is not None and r[0] == e[2] and r[1] == e[3] and r[2] == e[4]:
                pas[a] += 1
    sess = []
    for s in range(len(var)):
        sess.append([acct[s], lreq[s], lres[s], sig[s]])
    accs = []
    for a in range(len(tot)):
        accs.append([tot[a], pas[a]])
    return [sess, accs]