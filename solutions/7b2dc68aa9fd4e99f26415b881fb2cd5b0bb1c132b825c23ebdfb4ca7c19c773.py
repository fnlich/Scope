def audit_boundaries(units, operations):
    from array import array

    n = len(units)

    left = array('i', [0])
    right = array('i', [0])
    size = array('i', [0])
    value = array('H', [0])
    priority = array('I', [0])
    fcnt = array('i', [0])
    rcnt = array('i', [0])
    first = array('H', [0])
    last = array('H', [0])
    rev = array('b', [0])

    seed = 2463534242

    for v in units:
        seed ^= (seed << 13) & 0xffffffff
        seed ^= seed >> 17
        seed ^= (seed << 5) & 0xffffffff
        seed &= 0xffffffff
        left.append(0)
        right.append(0)
        size.append(1)
        value.append(v)
        priority.append(seed)
        fcnt.append(1)
        rcnt.append(1)
        first.append(v)
        last.append(v)
        rev.append(0)

    root = 0

    if n:
        stack = []
        for i in range(1, n + 1):
            x = 0
            p = priority[i]
            while stack and priority[stack[-1]] < p:
                x = stack.pop()
            left[i] = x
            if stack:
                right[stack[-1]] = i
            else:
                root = i
            stack.append(i)

        order = [root]
        j = 0
        while j < len(order):
            t = order[j]
            j += 1
            l = left[t]
            r = right[t]
            if l:
                order.append(l)
            if r:
                order.append(r)

        for t in reversed(order):
            l = left[t]
            r = right[t]
            v = value[t]

            if l:
                c = fcnt[l] + 1
                if 0xD800 <= last[l] <= 0xDBFF and 0xDC00 <= v <= 0xDFFF:
                    c -= 1
                ff = first[l]
                ll = v
            else:
                c = 1
                ff = v
                ll = v

            if r:
                c += fcnt[r]
                if 0xD800 <= ll <= 0xDBFF and 0xDC00 <= first[r] <= 0xDFFF:
                    c -= 1
                ll = last[r]

            fcnt[t] = c
            first[t] = ff
            last[t] = ll

            if r:
                c = rcnt[r]
                ff = last[r]
                ll = first[r]
                c += 1
                if 0xD800 <= first[r] <= 0xDBFF and 0xDC00 <= v <= 0xDFFF:
                    c -= 1
            else:
                c = 1
                ff = v
                ll = v

            if l:
                c += rcnt[l]
                if 0xD800 <= v <= 0xDBFF and 0xDC00 <= last[l] <= 0xDFFF:
                    c -= 1
                ll = first[l]

            rcnt[t] = c

    def apply_reverse(t):
        if not t:
            return
        left[t], right[t] = right[t], left[t]
        first[t], last[t] = last[t], first[t]
        fcnt[t], rcnt[t] = rcnt[t], fcnt[t]
        rev[t] ^= 1

    def push(t):
        if rev[t]:
            l = left[t]
            r = right[t]
            if l:
                apply_reverse(l)
            if r:
                apply_reverse(r)
            rev[t] = 0

    def pull(t):
        l = left[t]
        r = right[t]
        v = value[t]

        if l:
            c = fcnt[l] + 1
            if 0xD800 <= last[l] <= 0xDBFF and 0xDC00 <= v <= 0xDFFF:
                c -= 1
            ff = first[l]
            ll = v
        else:
            c = 1
            ff = v
            ll = v

        if r:
            c += fcnt[r]
            if 0xD800 <= ll <= 0xDBFF and 0xDC00 <= first[r] <= 0xDFFF:
                c -= 1
            ll = last[r]

        fcnt[t] = c
        first[t] = ff
        last[t] = ll

        if r:
            c = rcnt[r]
            ff = last[r]
            ll = first[r]
            c += 1
            if 0xD800 <= first[r] <= 0xDBFF and 0xDC00 <= v <= 0xDFFF:
                c -= 1
        else:
            c = 1
            ff = v
            ll = v

        if l:
            c += rcnt[l]
            if 0xD800 <= v <= 0xDBFF and 0xDC00 <= last[l] <= 0xDFFF:
                c -= 1
            ll = first[l]

        rcnt[t] = c

    def split(t, k):
        if not t:
            return 0, 0
        push(t)
        l = left[t]
        ls = size[l]
        if k <= ls:
            a, b = split(l, k)
            left[t] = b
            size[t] = size[left[t]] + size[right[t]] + 1
            pull(t)
            return a, t
        a, b = split(right[t], k - ls - 1)
        right[t] = a
        size[t] = size[left[t]] + size[right[t]] + 1
        pull(t)
        return t, b

    def merge(a, b):
        if not a:
            return b
        if not b:
            return a
        if priority[a] >= priority[b]:
            push(a)
            right[a] = merge(right[a], b)
            size[a] = size[left[a]] + size[right[a]] + 1
            pull(a)
            return a
        push(b)
        left[b] = merge(a, left[b])
        size[b] = size[left[b]] + size[right[b]] + 1
        pull(b)
        return b

    def set_at(t, pos, v):
        path = []
        while t:
            push(t)
            path.append(t)
            l = left[t]
            ls = size[l]
            if pos == ls:
                value[t] = v
                break
            if pos < ls:
                t = l
            else:
                pos -= ls + 1
                t = right[t]
        for t in reversed(path):
            pull(t)

    def inspect(t, pos):
        acc_count = 0
        acc_last = 0
        have = False

        while t:
            push(t)
            l = left[t]
            ls = size[l]

            if pos < ls:
                t = l
                continue

            if l:
                if have:
                    c = fcnt[l]
                    if 0xD800 <= acc_last <= 0xDBFF and 0xDC00 <= first[l] <= 0xDFFF:
                        c -= 1
                    acc_count += c
                else:
                    acc_count = fcnt[l]
                    have = True
                acc_last = last[l]

            if pos == ls:
                return (acc_last if have else -1), value[t], acc_count

            if have:
                if not (0xD800 <= acc_last <= 0xDBFF and 0xDC00 <= value[t] <= 0xDFFF):
                    acc_count += 1
            else:
                acc_count = 1
                have = True
            acc_last = value[t]
            pos -= ls + 1
            t = right[t]

        return -1, -1, acc_count

    def kth_start(t, k):
        base = 0
        prev = 0
        have_prev = False

        while t:
            push(t)
            l = left[t]
            ls = size[l]

            if l:
                cnt = fcnt[l]
                if have_prev and 0xD800 <= prev <= 0xDBFF and 0xDC00 <= first[l] <= 0xDFFF:
                    cnt -= 1
            else:
                cnt = 0

            if k < cnt:
                t = l
                continue

            k -= cnt
            v = value[t]
            is_start = not (
                have_prev and
                0xD800 <= prev <= 0xDBFF and
                0xDC00 <= v <= 0xDFFF
            )

            if is_start:
                if k == 0:
                    return base + ls
                k -= 1

            base += ls + 1
            prev = v
            have_prev = True
            t = right[t]

        return n

    reports = []

    for op in operations:
        typ = op[0]

        if typ == "set":
            set_at(root, op[1], op[2])

        elif typ == "reverse":
            lpos = op[1]
            rpos = op[2]
            if lpos < rpos - 1:
                a, bc = split(root, lpos)
                b, c = split(bc, rpos - lpos)
                apply_reverse(b)
                root = merge(a, merge(b, c))

        elif typ == "rotate":
            lpos = op[1]
            rpos = op[2]
            length = rpos - lpos
            if length > 1:
                k = op[3] % length
                if k:
                    a, bc = split(root, lpos)
                    b, c = split(bc, length)
                    x, y = split(b, length - k)
                    b = merge(y, x)
                    root = merge(a, merge(b, c))

        elif typ == "query":
            i = op[1]
            prev, cur, _ = inspect(root, i)
            if 0xD800 <= prev <= 0xDBFF and 0xDC00 <= cur <= 0xDFFF:
                reports.append(i - 1)
            else:
                reports.append(i)

        else:
            i = op[1]
            distance = op[2]
            prev, cur, starts_before = inspect(root, i)

            if 0xD800 <= prev <= 0xDBFF and 0xDC00 <= cur <= 0xDFFF:
                rank = starts_before - 1
            else:
                rank = starts_before

            total = fcnt[root]
            target = rank + distance
            if target < 0:
                target = 0
            elif target > total:
                target = total

            if target == total:
                reports.append(n)
            else:
                reports.append(kth_start(root, target))

    return reports