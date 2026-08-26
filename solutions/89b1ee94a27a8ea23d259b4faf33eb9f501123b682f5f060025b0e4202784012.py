def hid_lifecycle(n, horizon, initial_deadline, min_deadline, max_deadline, defer, sleeps, events):
    import heapq
    from bisect import bisect_left, bisect_right

    m = len(sleeps)
    starts = [0] * m
    ends = [0] * m
    awake_starts = [0] * m
    cum = [0] * (m + 1)

    for i, (s, e) in enumerate(sleeps):
        starts[i] = s
        ends[i] = e
        awake_starts[i] = s - cum[i]
        cum[i + 1] = cum[i] + e - s

    def awake_at(t):
        k = bisect_right(ends, t)
        return t - cum[k]

    def wall_at(a):
        k = bisect_left(awake_starts, a)
        return a + cum[k]

    initial = max(min_deadline, min(max_deadline, initial_deadline))

    CONNECTED = 0
    PENDING = 1
    DISCONNECTED = 2
    CONNECTING = 3

    state = [CONNECTED] * n
    request = [-1] * n
    watchdog_token = [0] * n
    release_token = [0] * n
    handshake_token = [0] * n

    heap = []
    result = []

    def schedule_watchdog(d, now_awake, duration):
        watchdog_token[d] += 1
        tok = watchdog_token[d]
        heapq.heappush(heap, (wall_at(now_awake + duration), d, 0, tok))

    def schedule_release(d, now_awake):
        release_token[d] += 1
        tok = release_token[d]
        heapq.heappush(heap, (wall_at(now_awake + defer), d, 1, tok))

    def schedule_handshake(d, now_awake, duration):
        handshake_token[d] += 1
        tok = handshake_token[d]
        heapq.heappush(heap, (wall_at(now_awake + duration), d, 2, tok))

    for d in range(n):
        schedule_watchdog(d, 0, initial)

    i = 0
    le = len(events)

    while i < le or heap:
        next_event_time = events[i][0] if i < le else None
        next_timer_time = heap[0][0] if heap else None

        if next_event_time is None:
            t = next_timer_time
        elif next_timer_time is None:
            t = next_event_time
        else:
            t = next_event_time if next_event_time <= next_timer_time else next_timer_time

        if t > horizon:
            break

        now_awake = awake_at(t)

        if next_event_time == t:
            while i < le and events[i][0] == t:
                _, d, kind, value = events[i]
                i += 1

                st = state[d]

                if st == CONNECTED:
                    if kind == "HB":
                        v = max(min_deadline, min(max_deadline, value))
                        schedule_watchdog(d, now_awake, v)
                    elif kind == "LOSS":
                        watchdog_token[d] += 1
                        state[d] = PENDING
                        request[d] = -1
                        schedule_release(d, now_awake)
                        result.append((t, d, "FAIL_LOSS"))

                elif st == PENDING:
                    if kind == "CONNECT":
                        request[d] = value

                elif st == DISCONNECTED:
                    if kind == "CONNECT":
                        state[d] = CONNECTING
                        schedule_handshake(d, now_awake, value)

                else:
                    if kind == "CONNECT":
                        schedule_handshake(d, now_awake, value)

        while heap and heap[0][0] == t:
            _, d, typ, tok = heapq.heappop(heap)

            if typ == 0:
                if tok != watchdog_token[d] or state[d] != CONNECTED:
                    continue
                watchdog_token[d] += 1
                state[d] = PENDING
                request[d] = -1
                schedule_release(d, now_awake)
                result.append((t, d, "FAIL_TIMEOUT"))

            elif typ == 1:
                if tok != release_token[d] or state[d] != PENDING:
                    continue
                release_token[d] += 1
                state[d] = DISCONNECTED
                result.append((t, d, "RELEASE"))
                req = request[d]
                request[d] = -1
                if req >= 0:
                    state[d] = CONNECTING
                    schedule_handshake(d, now_awake, req)

            else:
                if tok != handshake_token[d] or state[d] != CONNECTING:
                    continue
                handshake_token[d] += 1
                state[d] = CONNECTED
                result.append((t, d, "CONNECT"))
                schedule_watchdog(d, now_awake, initial)

    return result