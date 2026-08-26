def audit_ocsp_trace(events, cache_capacity):
    from collections import OrderedDict

    cache = OrderedDict()
    active = {}
    seen = set()

    RESOLVE_START = 0
    RESOLVE_DONE = 1
    ATTEMPT = 2
    ATTEMPT_RESULT = 3
    SENT_RESULT = 4
    CLOSE_FAIL = 5
    CLOSE_RESPONSE = 6
    RESET = 7
    CLEANUP = 8
    RESUME = 9
    CACHE_HIT = 10
    CACHE = 11

    for i, e in enumerate(events):
        tag = e[0]

        if tag == "BEGIN":
            _, rid, key, hostname, resolver, fallback_count = e
            if rid in seen:
                return (i, "INVALID")
            seen.add(rid)

            if key in cache:
                status = cache[key]
                cache.move_to_end(key)
                active[rid] = [key, fallback_count, hostname, resolver, -1, 0, CACHE_HIT, status]
            else:
                if hostname and resolver:
                    active[rid] = [key, fallback_count, hostname, resolver, -1, 0, RESOLVE_START, None]
                else:
                    total = fallback_count
                    stage = ATTEMPT if total else CLEANUP
                    active[rid] = [key, fallback_count, hostname, resolver, total, 0, stage, None]
            continue

        parts = e
        if len(parts) < 2:
            return (i, "INVALID")
        rid = parts[1]
        st = active.get(rid)
        if st is None:
            return (i, "INVALID")

        stage = st[6]

        if stage == CACHE_HIT:
            if tag != "CACHE_HIT" or parts[2] != st[7]:
                return (i, "INVALID")
            st[6] = CLEANUP

        elif stage == RESOLVE_START:
            if tag != "RESOLVE_START":
                return (i, "INVALID")
            st[6] = RESOLVE_DONE

        elif stage == RESOLVE_DONE:
            if tag != "RESOLVE_DONE":
                return (i, "INVALID")
            count = parts[2]
            st[4] = count + st[1]
            st[5] = 0
            st[6] = ATTEMPT if st[4] else CLEANUP

        elif stage == ATTEMPT:
            if tag != "ATTEMPT" or parts[2] != st[5]:
                return (i, "INVALID")
            st[6] = ATTEMPT_RESULT

        elif stage == ATTEMPT_RESULT:
            if tag == "FAIL":
                st[6] = CLOSE_FAIL
            elif tag == "SEND":
                st[6] = SENT_RESULT
            else:
                return (i, "INVALID")

        elif stage == SENT_RESULT:
            if tag == "FAIL":
                st[6] = CLOSE_FAIL
            elif tag == "RESPONSE":
                status = parts[2]
                st[7] = status
                st[6] = CLOSE_RESPONSE
            else:
                return (i, "INVALID")

        elif stage == CLOSE_FAIL:
            if tag != "CLOSE":
                return (i, "INVALID")
            st[5] += 1
            if st[5] < st[4]:
                st[6] = RESET
            else:
                st[6] = CLEANUP

        elif stage == CLOSE_RESPONSE:
            if tag != "CLOSE":
                return (i, "INVALID")
            if st[7] == "INVALID":
                st[6] = CLEANUP
            else:
                st[6] = CACHE

        elif stage == RESET:
            if tag != "RESET":
                return (i, "INVALID")
            st[6] = ATTEMPT

        elif stage == CACHE:
            if tag != "CACHE" or parts[2] != st[7]:
                return (i, "INVALID")
            key = st[0]
            cache[key] = st[7]
            cache.move_to_end(key)
            if len(cache) > cache_capacity:
                cache.popitem(last=False)
            st[6] = CLEANUP

        elif stage == CLEANUP:
            if tag != "CLEANUP":
                return (i, "INVALID")
            st[6] = RESUME

        elif stage == RESUME:
            if tag != "RESUME" or parts[2] != ("ERROR" if st[7] == "INVALID" else st[7]):
                return (i, "INVALID")
            del active[rid]

        else:
            return (i, "INVALID")

    if active:
        return (len(events), "INCOMPLETE")
    return (-1, "VALID")