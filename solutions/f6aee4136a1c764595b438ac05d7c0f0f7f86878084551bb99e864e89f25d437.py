import json
import heapq


def simulate_resolution(requests, durations, limit):
    dur = {}
    for k, v in durations.items():
        if isinstance(k, str):
            kk = None
            try:
                kk = json.loads(k)
            except Exception:
                kk = None
            if isinstance(kk, (list, tuple)) and len(kk) == 2:
                dur[(kk[0], kk[1])] = v
            else:
                dur[k] = v
        elif isinstance(k, (list, tuple)) and len(k) == 2:
            dur[(k[0], k[1])] = v
        else:
            dur[k] = v

    hexd = set("0123456789abcdefABCDEF")
    zonech = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    lblch = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

    def is_v4(s):
        parts = s.split(".")
        if len(parts) != 4:
            return False
        for p in parts:
            if len(p) < 1 or len(p) > 3:
                return False
            for c in p:
                if c < "0" or c > "9":
                    return False
            if len(p) > 1 and p[0] == "0":
                return False
            if int(p) > 255:
                return False
        return True

    def is_v6(s):
        if "%" in s:
            i = s.index("%")
            zone = s[i + 1:]
            s = s[:i]
            if len(zone) < 1 or len(zone) > 15:
                return False
            for c in zone:
                if c not in zonech:
                    return False
            if "%" in s:
                return False
        if s == "":
            return False
        if "::" in s:
            i = s.index("::")
            head = s[:i]
            tail = s[i + 2:]
            if "::" in tail:
                return False
            hp = head.split(":") if head != "" else []
            tp = tail.split(":") if tail != "" else []
            compressed = True
        else:
            hp = s.split(":")
            tp = []
            compressed = False
        parts = hp + tp
        for p in parts:
            if p == "":
                return False
        n4 = 0
        if parts and "." in parts[-1]:
            if not is_v4(parts[-1]):
                return False
            parts = parts[:-1]
            n4 = 2
        for p in parts:
            if len(p) < 1 or len(p) > 4:
                return False
            for c in p:
                if c not in hexd:
                    return False
        cnt = len(parts) + n4
        if compressed:
            return cnt <= 7
        return cnt == 8

    def dns_norm(name):
        s = name
        if s.endswith("."):
            s = s[:-1]
        s = s.lower()
        if len(s) < 1 or len(s) > 253:
            return None
        for lab in s.split("."):
            if len(lab) < 1 or len(lab) > 63:
                return None
            if lab[0] == "-" or lab[-1] == "-":
                return None
            for c in lab:
                if c not in lblch:
                    return None
        return s

    n = len(requests)
    outcomes = [None] * n
    req_key = [None] * n
    arrivals = {}
    cancels = {}
    times = []
    seen_times = set()

    def push_time(t):
        if t not in seen_times:
            seen_times.add(t)
            heapq.heappush(times, t)

    for i in range(n):
        r = requests[i]
        arrival = r[0]
        family = r[1]
        name = r[2]
        cancel = r[3]
        if name == "":
            outcomes[i] = ("error", "EMPTY")
            continue
        if family not in (0, 4, 6):
            outcomes[i] = ("error", "INVALID_FAMILY")
            continue
        if is_v4(name):
            if family == 6:
                outcomes[i] = ("error", "ADDRESS_FAMILY")
            else:
                outcomes[i] = ("literal", name)
            continue
        if is_v6(name):
            if family == 4:
                outcomes[i] = ("error", "ADDRESS_FAMILY")
            else:
                outcomes[i] = ("literal", name)
            continue
        nm = dns_norm(name)
        if nm is None:
            outcomes[i] = ("error", "INVALID_NAME")
            continue
        req_key[i] = (family, nm)
        if arrival in arrivals:
            arrivals[arrival].append(i)
        else:
            arrivals[arrival] = [i]
        push_time(arrival)
        if cancel is not None:
            if cancel in cancels:
                cancels[cancel].append(i)
            else:
                cancels[cancel] = [i]
            push_time(cancel)

    jobs = []
    job_members = {}
    job_key = {}
    running_by_key = {}
    completions = {}
    groups = {}
    group_heap = []
    order = 0
    caller_loc = {}

    def create_job(t, key, members):
        jid = len(jobs)
        d = dur.get(key, 1)
        stop = t + d
        jobs.append([jid, key[0], key[1], t, stop, "completed"])
        job_members[jid] = set(members)
        job_key[jid] = key
        running_by_key[key] = jid
        for m in members:
            caller_loc[m] = (0, jid)
        if stop in completions:
            completions[stop].append(jid)
        else:
            completions[stop] = [jid]
        push_time(stop)
        return jid

    def start_queued(t):
        while len(running_by_key) < limit and group_heap:
            o, key = heapq.heappop(group_heap)
            if key not in groups:
                continue
            members = groups.pop(key)
            create_job(t, key, members)

    while times:
        t = heapq.heappop(times)
        seen_times.discard(t)

        clist = completions.pop(t, None)
        if clist:
            clist.sort()
            for jid in clist:
                if jid not in job_members:
                    continue
                members = job_members.pop(jid)
                key = job_key[jid]
                if running_by_key.get(key) == jid:
                    del running_by_key[key]
                jobs[jid][4] = t
                jobs[jid][5] = "completed"
                for i in members:
                    outcomes[i] = ("resolved", jid, t)

        start_queued(t)

        alist = arrivals.pop(t, None)
        if alist:
            for i in alist:
                key = req_key[i]
                if key in running_by_key:
                    jid = running_by_key[key]
                    job_members[jid].add(i)
                    caller_loc[i] = (0, jid)
                elif key in groups:
                    groups[key].add(i)
                    caller_loc[i] = (1, key)
                elif len(running_by_key) < limit:
                    create_job(t, key, [i])
                else:
                    groups[key] = set([i])
                    heapq.heappush(group_heap, (order, key))
                    order += 1
                    caller_loc[i] = (1, key)

        xlist = cancels.pop(t, None)
        if xlist:
            for i in xlist:
                if outcomes[i] is not None:
                    continue
                loc = caller_loc.get(i)
                if loc is None:
                    continue
                if loc[0] == 0:
                    jid = loc[1]
                    if jid not in job_members:
                        continue
                    s = job_members[jid]
                    s.discard(i)
                    outcomes[i] = ("canceled", t)
                    if not s:
                        del job_members[jid]
                        key = job_key[jid]
                        if running_by_key.get(key) == jid:
                            del running_by_key[key]
                        jobs[jid][4] = t
                        jobs[jid][5] = "canceled"
                else:
                    key = loc[1]
                    g = groups.get(key)
                    if g is None:
                        continue
                    g.discard(i)
                    outcomes[i] = ("canceled", t)
                    if not g:
                        del groups[key]

        start_queued(t)

    return (outcomes, [tuple(j) for j in jobs])