def channel_calendar(events, until):
    import heapq

    records = {}
    owned = {}
    watchers = {}
    closed = set()
    actions = []
    seq = 0

    def options_dict(pairs):
        return {k: v for k, v in pairs}

    def merged_options(r):
        d = dict(r["static"])
        d.update(r["dynamic"])
        return tuple(sorted(d.items()))

    def remove_record(key):
        r = records.pop(key, None)
        if r is None:
            return
        if r["owner"] is not None:
            s = owned.get(r["owner"])
            if s is not None:
                s.discard(key)
                if not s:
                    owned.pop(r["owner"], None)
        r["pending"] = None

    def install_record(key, channel, participant, process, token, state, static, dynamic):
        remove_record(key)
        r = {
            "channel": channel,
            "participant": participant,
            "owner": process,
            "token": token,
            "state": state,
            "static": options_dict(static),
            "dynamic": options_dict(dynamic),
            "pending": None,
            "version": 0,
        }
        records[key] = r
        owned.setdefault(process, set()).add(key)

    def close_channel(channel):
        if channel in closed:
            return
        closed.add(channel)
        watchers.pop(channel, None)
        keys = [k for k in records if k[0] == channel]
        for k in keys:
            remove_record(k)

    def process_actions(limit):
        nonlocal actions
        while actions and actions[0][0] <= limit:
            due, kind, s, data = heapq.heappop(actions)
            if kind == 0:
                channel, wid = data
                w = watchers.get(channel)
                if w is not None and w[0] == wid and w[1] == due:
                    close_channel(channel)
            else:
                key, r, ver, token = data
                cur = records.get(key)
                if cur is r and cur["version"] == ver and cur["pending"] is not None:
                    ptoken, pdue = cur["pending"]
                    if pdue == due and ptoken == token:
                        cur["token"] = token
                        cur["pending"] = None

    indexed = []
    for i, e in enumerate(events):
        if len(e) >= 2 and e[1] <= until:
            indexed.append((e[1], i, e))
    indexed.sort(key=lambda x: (x[0], x[1]))

    watcher_id = 0

    for t, _, e in indexed:
        process_actions(t)
        kind = e[0]

        if kind == "PUT":
            _, _, channel, participant, process, token, state, static, dynamic = e
            if channel not in closed:
                install_record(
                    (channel, participant),
                    channel, participant, process, token, state, static, dynamic
                )

        elif kind == "EDIT":
            _, _, channel, participant, state, dynamic = e
            if channel not in closed:
                r = records.get((channel, participant))
                if r is not None:
                    r["state"] = state
                    r["dynamic"] = options_dict(dynamic)

        elif kind == "ROTATE":
            _, _, channel, participant, token, delay = e
            if channel not in closed:
                key = (channel, participant)
                r = records.get(key)
                if r is not None:
                    r["version"] += 1
                    due = t + delay
                    r["pending"] = (token, due)
                    seq += 1
                    heapq.heappush(actions, (due, 1, seq, (key, r, r["version"], token)))

        elif kind == "CRASH":
            _, _, process = e
            if process in owned:
                keys = list(owned[process])
                for key in keys:
                    r = records.get(key)
                    if r is None or r["owner"] != process:
                        continue
                    r["owner"] = None
                    r["static"] = dict(
                        tuple(r["static"].items()) + tuple(r["dynamic"].items())
                    )
                    r["dynamic"] = {}
                    r["pending"] = None
                    r["version"] += 1
                    owned[process].discard(key)
                if not owned[process]:
                    owned.pop(process, None)

        elif kind == "RESTORE":
            _, _, channel, participant, process, token = e
            if channel not in closed:
                key = (channel, participant)
                r = records.get(key)
                if r is not None and r["owner"] is None and r["token"] == token:
                    r["owner"] = process
                    r["dynamic"] = {}
                    owned.setdefault(process, set()).add(key)

        elif kind == "PLAN":
            _, _, channel, delay = e
            if channel not in closed and channel not in watchers:
                watcher_id += 1
                due = t + delay
                watchers[channel] = (watcher_id, due)
                seq += 1
                heapq.heappush(actions, (due, 0, seq, (channel, watcher_id)))

        elif kind == "CANCEL":
            _, _, channel = e
            if channel not in closed:
                watchers.pop(channel, None)

    process_actions(until)

    out_records = []
    for key in sorted(records):
        r = records[key]
        rotation = r["pending"]
        out_records.append((
            r["channel"],
            r["participant"],
            "memory" if r["owner"] is not None else "durable",
            r["state"],
            r["token"],
            merged_options(r),
            r["owner"],
            rotation
        ))

    monitors = sorted(
        (process, channel, participant)
        for process, keys in owned.items()
        for channel, participant in keys
    )

    pending_watchers = sorted(
        (channel, due) for channel, (_, due) in watchers.items()
    )

    return out_records, monitors, pending_watchers