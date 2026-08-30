def solve_terminal_editor(width, primary_prompt, continuation_prompt, initial, operations):
    W = width
    cont = continuation_prompt
    M = W + 1
    left = []
    right = list(initial)
    right.reverse()
    states = [primary_prompt]
    res = []
    for op in operations:
        t = op[0]
        if t == "I":
            w = op[1]
            left.append(w)
            del states[len(left):]
        elif t == "B":
            if left:
                left.pop()
                del states[len(left) + 1:]
        elif t == "D":
            if right:
                right.pop()
                del states[len(left) + 1:]
        elif t == "L":
            k = op[1]
            m = len(left)
            if k < m:
                m = k
            if m > 0:
                seg = left[len(left) - m:]
                seg.reverse()
                right.extend(seg)
                del left[len(left) - m:]
        elif t == "R":
            k = op[1]
            m = len(right)
            if k < m:
                m = k
            if m > 0:
                seg = right[len(right) - m:]
                seg.reverse()
                left.extend(seg)
                del right[len(right) - m:]
        elif t == "V":
            k = op[1]
            m = len(right)
            if k < m:
                m = k
            if m >= 2:
                start = len(right) - m
                seg = right[start:]
                seg.reverse()
                right[start:] = seg
                del states[len(left) + 1:]
        c = len(left)
        ls = len(states)
        if ls <= c:
            e = states[ls - 1]
            row = e // M
            col = e - row * M
            ap = states.append
            for i in range(ls - 1, c):
                w = left[i]
                nc = col + w
                if nc <= W:
                    col = nc
                else:
                    row += 1
                    col = cont + w
                ap(row * M + col)
        else:
            e = states[c]
            row = e // M
            col = e - row * M
        res.append((c, row, col))
    return res