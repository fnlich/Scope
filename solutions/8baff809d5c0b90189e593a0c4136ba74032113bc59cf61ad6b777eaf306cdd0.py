def infer_shapes(tensors, relations, observations, caps, alignments, scalings):
    from fractions import Fraction
    from math import gcd

    names = {}
    leaves = []

    for shape in tensors:
        out = []
        for x in shape:
            if isinstance(x, str):
                if x not in names:
                    names[x] = len(names)
                out.append(names[x])
            else:
                out.append(x)
        leaves.append(out)

    n = len(names)
    if n == 0:
        result = []
        for shape in tensors:
            result.append([["exact", x] for x in shape])
        return {"status": "ok", "shapes": result}

    adj = [[] for _ in range(n)]

    for x, y, d in relations:
        ix = names[x]
        iy = names[y]
        adj[iy].append((ix, 1, 1, d, 1))
        adj[ix].append((iy, 1, 1, -d, 1))

    for x, y, m, d in scalings:
        ix = names[x]
        iy = names[y]
        adj[iy].append((ix, m, 1, d, 1))
        adj[ix].append((iy, 1, m, -d, m))

    aa = [None] * n
    bb = [None] * n
    comp = [-1] * n
    components = []
    exact = []

    bad = False

    for start in range(n):
        if comp[start] != -1:
            continue
        ci = len(components)
        components.append([])
        exact.append(None)
        stack = [start]
        comp[start] = ci
        aa[start] = Fraction(1)
        bb[start] = Fraction(0)

        while stack:
            u = stack.pop()
            components[ci].append(u)
            au = aa[u]
            bu = bb[u]
            for v, kn, kd, dn, dd in adj[u]:
                k = Fraction(kn, kd)
                d = Fraction(dn, dd)
                na = k * au
                nb = k * bu + d
                if comp[v] == -1:
                    comp[v] = ci
                    aa[v] = na
                    bb[v] = nb
                    stack.append(v)
                else:
                    ca = aa[v] - na
                    cb = nb - bb[v]
                    if ca == 0:
                        if cb != 0:
                            bad = True
                            break
                    else:
                        t = cb / ca
                        if exact[ci] is None:
                            exact[ci] = t
                        elif exact[ci] != t:
                            bad = True
                            break
            if bad:
                break
        if bad:
            break

    if bad:
        return {"status": "contradiction"}

    def add_exact(ci, t):
        nonlocal bad
        if exact[ci] is None:
            exact[ci] = t
        elif exact[ci] != t:
            bad = True

    for x, value in observations:
        i = names[x]
        ci = comp[i]
        if exact[ci] is not None:
            continue
        a = aa[i]
        b = bb[i]
        if a == 0:
            if b != value:
                bad = True
                break
        else:
            add_exact(ci, (Fraction(value) - b) / a)

    if bad:
        return {"status": "contradiction"}

    for ci in range(len(components)):
        t = exact[ci]
        if t is not None and t.denominator != 1:
            return {"status": "contradiction"}

    lo = [1] * len(components)
    hi = [None] * len(components)
    cr = [0] * len(components)
    cm = [1] * len(components)

    def flo(x):
        return x.numerator // x.denominator

    def cei(x):
        return -((-x.numerator) // x.denominator)

    def merge(ci, r, m):
        nonlocal bad
        if m == 1:
            return
        r %= m
        oldr = cr[ci]
        oldm = cm[ci]
        g = gcd(oldm, m)
        if (r - oldr) % g:
            bad = True
            return
        m2 = m // g
        if m2 == 1:
            k = 0
        else:
            k = ((r - oldr) // g) * pow(oldm // g, -1, m2) % m2
        nm = oldm * m2
        nr = (oldr + oldm * k) % nm
        cr[ci] = nr
        cm[ci] = nm

    def affine_congruence(a, b, target_mod, target_rem):
        d1 = a.denominator
        d2 = b.denominator
        g = gcd(d1, d2)
        d = d1 // g * d2
        p = a.numerator * (d // d1)
        q = b.numerator * (d // d2) - d * target_rem
        mod = d * target_mod
        gg = gcd(p, mod)
        if q % gg:
            return None
        mm = mod // gg
        if mm == 1:
            return (0, 1)
        pp = p // gg
        qq = -q // gg
        r = (qq * pow(pp % mm, -1, mm)) % mm
        return (r, mm)

    for i in range(n):
        ci = comp[i]
        if exact[ci] is not None:
            continue

        a = aa[i]
        b = bb[i]

        if a == 0:
            if b.denominator != 1 or b < 1:
                bad = True
                break
        else:
            d1 = a.denominator
            d2 = b.denominator
            g = gcd(d1, d2)
            d = d1 // g * d2
            p = a.numerator * (d // d1)
            q = b.numerator * (d // d2)

            gg = gcd(abs(p), d)
            if q % gg:
                bad = True
                break
            mm = d // gg
            if mm != 1:
                rhs = (-q // gg) % mm
                pp = (p // gg) % mm
                r = rhs * pow(pp, -1, mm) % mm
                merge(ci, r, mm)
                if bad:
                    break

            if a > 0:
                nl = cei((Fraction(1) - b) / a)
                if nl > lo[ci]:
                    lo[ci] = nl
            else:
                nh = flo((Fraction(1) - b) / a)
                if hi[ci] is None or nh < hi[ci]:
                    hi[ci] = nh

            if b.denominator != 1 or a == 0:
                pass

    if bad:
        return {"status": "contradiction"}

    for x, upper in caps:
        i = names[x]
        ci = comp[i]
        if exact[ci] is not None:
            continue
        a = aa[i]
        b = bb[i]
        if a == 0:
            if b > upper:
                bad = True
                break
        elif a > 0:
            nh = flo((Fraction(upper) - b) / a)
            if hi[ci] is None or nh < hi[ci]:
                hi[ci] = nh
        else:
            nl = cei((Fraction(upper) - b) / a)
            if nl > lo[ci]:
                lo[ci] = nl

    if bad:
        return {"status": "contradiction"}

    for x, modulus, remainder in alignments:
        i = names[x]
        ci = comp[i]
        if exact[ci] is not None:
            continue
        a = aa[i]
        b = bb[i]
        if a == 0:
            if b.denominator != 1 or b.numerator % modulus != remainder:
                bad = True
                break
        else:
            c = affine_congruence(a, b, modulus, remainder)
            if c is None:
                bad = True
                break
            merge(ci, c[0], c[1])
            if bad:
                break

    if bad:
        return {"status": "contradiction"}

    for ci in range(len(components)):
        t = exact[ci]
        if t is not None:
            ti = t.numerator
            for i in components[ci]:
                v = aa[i] * ti + bb[i]
                if v.denominator != 1 or v < 1:
                    return {"status": "contradiction"}
            continue

        m = cm[ci]
        r = cr[ci]
        l = lo[ci]
        h = hi[ci]

        first = r + ((l - r + m - 1) // m) * m
        if h is not None:
            if first > h:
                return {"status": "contradiction"}
            last = r + ((h - r) // m) * m
            if last < first:
                return {"status": "contradiction"}

    for x, value in observations:
        i = names[x]
        ci = comp[i]
        t = exact[ci]
        if t is not None:
            v = aa[i] * t + bb[i]
            if v != value:
                return {"status": "contradiction"}

    for x, upper in caps:
        i = names[x]
        ci = comp[i]
        t = exact[ci]
        if t is not None:
            v = aa[i] * t + bb[i]
            if v > upper:
                return {"status": "contradiction"}

    for x, modulus, remainder in alignments:
        i = names[x]
        ci = comp[i]
        t = exact[ci]
        if t is not None:
            v = aa[i] * t + bb[i]
            if v.denominator != 1 or v.numerator % modulus != remainder:
                return {"status": "contradiction"}

    comp_first = [None] * len(components)
    comp_last = [None] * len(components)

    for ci in range(len(components)):
        t = exact[ci]
        if t is not None:
            comp_first[ci] = t.numerator
            comp_last[ci] = t.numerator
        else:
            m = cm[ci]
            r = cr[ci]
            l = lo[ci]
            h = hi[ci]
            f = r + ((l - r + m - 1) // m) * m
            comp_first[ci] = f
            if h is not None:
                comp_last[ci] = r + ((h - r) // m) * m

    info = [None] * n

    for i in range(n):
        ci = comp[i]
        t = exact[ci]
        if t is not None:
            v = aa[i] * t + bb[i]
            info[i] = ["exact", v.numerator]
            continue

        a = aa[i]
        b = bb[i]
        first_t = comp_first[ci]
        last_t = comp_last[ci]
        m = cm[ci]

        if a == 0:
            if b.denominator != 1 or b < 1:
                return {"status": "contradiction"}
            info[i] = ["exact", b.numerator]
            continue

        step = abs(a * m)
        if step.denominator != 1:
            return {"status": "contradiction"}
        step = step.numerator

        v1 = a * first_t + b
        if v1.denominator != 1 or v1 < 1:
            return {"status": "contradiction"}
        v1 = v1.numerator

        if last_t is None:
            if a <= 0:
                return {"status": "contradiction"}
            info[i] = ["unbounded", v1, step]
        else:
            v2 = a * last_t + b
            if v2.denominator != 1 or v2 < 1:
                return {"status": "contradiction"}
            v2 = v2.numerator
            smallest = min(v1, v2)
            largest = max(v1, v2)
            if smallest == largest:
                info[i] = ["exact", smallest]
            else:
                info[i] = ["bounded", smallest, largest, step]

    result = []
    for shape in leaves:
        result.append([info[x][:] if isinstance(x, int) and x < n and info[x] is not None else ["exact", x] for x in shape])
    return {"status": "ok", "shapes": result}