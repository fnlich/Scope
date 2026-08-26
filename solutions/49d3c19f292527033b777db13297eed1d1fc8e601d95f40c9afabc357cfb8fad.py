def normalize_gps_log(records, max_gap_seconds):
    from datetime import date
    from bisect import bisect_left
    from math import gcd

    DAY = 86400000
    G = max_gap_seconds * 1000
    n = len(records)

    def text(v):
        if isinstance(v, bytes):
            try:
                s = v.decode("utf-8")
            except UnicodeDecodeError:
                return None
        elif isinstance(v, str):
            s = v
        else:
            return None
        s = s.rstrip("\x00")
        for c in s:
            o = ord(c)
            if o < 32 or o > 126:
                return None
        return s.strip(" ")

    def rat(v):
        if not isinstance(v, list) or len(v) != 2:
            return None
        a, b = v
        if isinstance(a, bool) or isinstance(b, bool):
            return None
        if not isinstance(a, int) or not isinstance(b, int) or b == 0:
            return None
        if b < 0:
            a = -a
            b = -b
        g = gcd(abs(a), b)
        return a // g, b // g

    def integer_rat(v):
        r = rat(v)
        if r is None:
            return None
        a, b = r
        if a % b:
            return None
        return a // b

    def coordinate(v, ref, is_lat):
        if not isinstance(v, list) or len(v) != 3:
            return None
        d = integer_rat(v[0])
        if d is None or d < 0:
            return None
        m = rat(v[1])
        s = rat(v[2])
        if m is None or s is None:
            return None
        mn, md = m
        sn, sd = s
        if mn < 0 or mn >= 60 * md or sn < 0 or sn >= 60 * sd:
            return None
        limit = 90 if is_lat else 180
        if d > limit:
            return None
        if d == limit and (mn != 0 or sn != 0):
            return None

        arc_num = d * 3600 * md * sd + mn * 60 * sd + sn * md
        arc_den = md * sd
        num = arc_num * 1000000
        den = arc_den * 3600
        rounded = (2 * num + den) // (2 * den)

        rtxt = text(ref)
        if rtxt is None:
            return None
        rtxt = rtxt.upper()
        if is_lat:
            if rtxt not in ("N", "S"):
                return None
            if rtxt == "S":
                rounded = -rounded
        else:
            if rtxt not in ("E", "W"):
                return None
            if rtxt == "W":
                rounded = -rounded

        sign = "-" if rounded < 0 else ""
        x = -rounded if rounded < 0 else rounded
        return sign + str(x // 1000000) + "." + f"{x % 1000000:06d}"

    def position(rec):
        try:
            lat = coordinate(rec["lat"], rec["lat_ref"], True)
            lon = coordinate(rec["lon"], rec["lon_ref"], False)
        except (KeyError, TypeError):
            return None
        if lat is None or lon is None:
            return None
        return lat + "," + lon

    def ordinal(y, m, d):
        y1 = y - 1
        z = 365 * y1 + y1 // 4 - y1 // 100 + y1 // 400
        leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        mdays = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335) if leap else (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
        return z + mdays[m - 1] + d - 1

    def parse_date(v):
        s = text(v)
        if s is None or len(s) != 10:
            return None
        if s[4] != ":" or s[7] != ":":
            return None
        for i in (0, 1, 2, 3, 5, 6, 8, 9):
            if s[i] < "0" or s[i] > "9":
                return None
        y = int(s[:4])
        m = int(s[5:7])
        d = int(s[8:10])
        if y < 1 or y > 9999 or m < 1 or m > 12:
            return None
        leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        ml = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        if d < 1 or d > ml[m - 1]:
            return None
        return ordinal(y, m, d)

    def parse_time(rec):
        try:
            tv = rec["time"]
        except (KeyError, TypeError):
            return None
        if not isinstance(tv, list) or len(tv) != 3:
            return None
        h = integer_rat(tv[0])
        mi = integer_rat(tv[1])
        sec = rat(tv[2])
        if h is None or mi is None or sec is None:
            return None
        if h < 0 or h > 23 or mi < 0 or mi > 59:
            return None
        sn, sd = sec
        if sn < 0 or sn >= 60 * sd:
            return None
        ms = (2 * sn * 1000 + sd) // (2 * sd)
        carry = 0
        if ms >= 60000:
            ms = 0
            carry = 1
        return h * 3600000 + mi * 60000 + ms, carry

    def parse_domain(rec, carry):
        if "date" not in rec or rec["date"] is None:
            return None, True
        v = rec["date"]
        if isinstance(v, list):
            if not v:
                return None, False
            vals = v
        else:
            vals = [v]
        out = set()
        for x in vals:
            d = parse_date(x)
            if d is None:
                return None, False
            d += carry
            if 0 <= d < 3652059:
                out.add(d)
        if not out:
            return None, False
        return sorted(out), True

    outputs = [None] * n
    valid_indices = []
    positions = []
    tods = []
    domains = []

    for i, rec in enumerate(records):
        p = position(rec)
        t = parse_time(rec)
        if t is None:
            outputs[i] = (p if p is not None else "INVALID_POSITION") + "|INVALID_TIME"
            continue
        tod, carry = t
        dom, ok = parse_domain(rec, carry)
        if not ok:
            outputs[i] = (p if p is not None else "INVALID_POSITION") + "|INVALID_TIME"
            continue
        valid_indices.append(i)
        positions.append(p if p is not None else "INVALID_POSITION")
        tods.append(tod)
        domains.append(dom)

    m = len(valid_indices)
    if m == 0:
        return outputs

    L = [0] * (m - 1)
    U = [0] * (m - 1)
    pref_l = [0] * m
    pref_u = [0] * m

    for i in range(m - 1):
        delta = tods[i + 1] - tods[i]
        x = 1 - delta
        l = -((-x) // DAY)
        u = (G - delta) // DAY
        L[i] = l
        U[i] = u
        pref_l[i + 1] = pref_l[i] + l
        pref_u[i + 1] = pref_u[i] + u

    constrained = [i for i in range(m) if domains[i] is not None]

    feasible = {}
    last_c = constrained[-1]
    suffix_l = pref_l[m - 1] - pref_l[last_c]
    last_domain = domains[last_c]
    if suffix_l:
        lim = 3652058 - suffix_l
        last_domain = [d for d in last_domain if d <= lim]
    feasible[last_c] = last_domain

    for cj in range(len(constrained) - 2, -1, -1):
        a = constrained[cj]
        b = constrained[cj + 1]
        A = pref_l[b] - pref_l[a]
        B = pref_u[b] - pref_u[a]
        nxt = feasible[b]
        cur = domains[a]
        good = []
        lo = 0
        ln = len(nxt)
        for d in cur:
            lo = bisect_left(nxt, d + A, lo)
            if lo < ln and nxt[lo] <= d + B:
                good.append(d)
        feasible[a] = good

    first_c = constrained[0]
    prefix_l = pref_l[first_c]
    first_good = [d for d in feasible[first_c] if d >= prefix_l]
    feasible[first_c] = first_good

    assignment = [None] * m

    if not first_good:
        for j in range(m):
            idx = valid_indices[j]
            outputs[idx] = positions[j] + "|TIME_CONFLICT"
        return outputs

    chosen = first_good[0]
    assignment[first_c] = chosen

    for ci in range(1, len(constrained)):
        a = constrained[ci - 1]
        b = constrained[ci]
        prev = assignment[a]
        A = pref_l[b] - pref_l[a]
        B = pref_u[b] - pref_u[a]
        dom = feasible[b]
        k = bisect_left(dom, prev + A)
        if k >= len(dom) or dom[k] > prev + B:
            for j in range(m):
                idx = valid_indices[j]
                outputs[idx] = positions[j] + "|TIME_CONFLICT"
            return outputs
        assignment[b] = dom[k]

        for j in range(a + 1, b):
            rem_l = pref_l[b] - pref_l[j]
            rem_u = pref_u[b] - pref_u[j]
            low = max(prev + L[j - 1], chosen if False else -10**30, assignment[b] - rem_u, 0)
            high = min(prev + U[j - 1], assignment[b] - rem_l, 3652058)
            if low > high:
                for q in range(m):
                    idx = valid_indices[q]
                    outputs[idx] = positions[q] + "|TIME_CONFLICT"
                return outputs
            assignment[j] = low
            prev = low

    first = first_c
    prev = assignment[first]
    for j in range(first - 1, -1, -1):
        pass

    if first > 0:
        end_date = assignment[first]
        prev = None
        for j in range(first):
            rem_l = pref_l[first] - pref_l[j]
            rem_u = pref_u[first] - pref_u[j]
            low = max(0, end_date - rem_u)
            high = min(3652058, end_date - rem_l)
            if prev is not None:
                low = max(low, prev + L[j - 1])
                high = min(high, prev + U[j - 1])
            if low > high:
                for q in range(m):
                    idx = valid_indices[q]
                    outputs[idx] = positions[q] + "|TIME_CONFLICT"
                return outputs
            assignment[j] = low
            prev = low

    last = constrained[-1]
    prev = assignment[last]
    for j in range(last + 1, m):
        low = max(0, prev + L[j - 1])
        high = min(3652058, prev + U[j - 1])
        if low > high:
            for q in range(m):
                idx = valid_indices[q]
                outputs[idx] = positions[q] + "|TIME_CONFLICT"
            return outputs
        assignment[j] = low
        prev = low

    for j in range(m):
        d = assignment[j]
        dt = date.fromordinal(d + 1)
        total = tods[j]
        hh = total // 3600000
        rem = total % 3600000
        mm = rem // 60000
        rem %= 60000
        ss = rem // 1000
        ms = rem % 1000
        timestamp = (
            f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
            f"T{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}Z"
        )
        outputs[valid_indices[j]] = positions[j] + "|" + timestamp

    return outputs