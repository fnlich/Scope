def render_viewport(markup, width, max_lines, align, tab_size):
    import unicodedata

    lines = []
    stack = None
    current = []
    sep = []

    def flush_sep():
        nonlocal sep
        if sep:
            current.append([1, sep])
            sep = []

    def fold(items, stop_after):
        out = []
        i = 0
        units = []
        col = 0
        opp = None

        while i < len(items):
            item = items[i]
            kind = item[0]

            if kind == 1:
                if i == len(items) - 1 or col == 0:
                    i += 1
                    continue

                x = col
                exp = 0
                for node, ch in item[1]:
                    if ch == ' ':
                        x += 1
                        exp += 1
                    else:
                        d = tab_size - (x % tab_size)
                        x += d
                        exp += d

                opp = (len(units), col, i + 1, False, None)

                if col + exp <= width:
                    for node, ch in item[1]:
                        if ch == ' ':
                            units.append([node, ' ', 1])
                            col += 1
                        else:
                            d = tab_size - (col % tab_size)
                            for _ in range(d):
                                units.append([node, ' ', 1])
                            col += d
                    i += 1
                else:
                    units = units[:opp[0]]
                    out.append(units)
                    if len(out) >= stop_after:
                        return out
                    units = []
                    col = 0
                    opp = None
                    i = opp[2] if opp is not None else i + 1
                    continue

            elif kind == 2:
                node = item[1]
                if col + 1 <= width:
                    opp = (len(units), col, i + 1, True, node)
                    i += 1
                else:
                    out.append(units)
                    if len(out) >= stop_after:
                        return out
                    units = []
                    col = 0
                    opp = None
                    i += 1

            else:
                node = item[1]
                text = item[2]
                w = item[3]

                if col + w <= width:
                    units.append([node, text, w])
                    col += w
                    i += 1
                elif opp is not None:
                    cp, cpcol, cont, is_hyphen, hnode = opp
                    units = units[:cp]
                    col = cpcol
                    if is_hyphen:
                        units.append([hnode, '-', 1])
                        col += 1
                    out.append(units)
                    if len(out) >= stop_after:
                        return out
                    units = []
                    col = 0
                    opp = None
                    i = cont
                elif col == 0 and w == 2 and width == 1:
                    units.append([node, '�', 1])
                    col = 1
                    i += 1
                else:
                    out.append(units)
                    if len(out) >= stop_after:
                        return out
                    units = []
                    col = 0
                    opp = None

        out.append(units)
        return out[:stop_after]

    def add_folded():
        nonlocal current
        remaining = max_lines + 1 - len(lines)
        if remaining <= 0:
            return True
        folded = fold(current, remaining)
        lines.extend(folded)
        current = []
        return len(lines) > max_lines

    n = len(markup)
    i = 0

    while i < n:
        c = markup[i]

        if c == '[':
            if i + 1 < n and markup[i + 1] == '[':
                flush_sep()
                current.append([0, stack, '[', 1])
                i += 2
            else:
                j = markup.find(']', i + 1)
                token = markup[i + 1:j]
                if token == '/':
                    stack = stack[0]
                else:
                    stack = (stack, token)
                i = j + 1

        elif c == '\n':
            flush_sep()
            if add_folded():
                break
            i += 1

        elif c == ' ' or c == '\t':
            sep.append((stack, c))
            i += 1

        elif c == '\u00ad':
            flush_sep()
            current.append([2, stack])
            i += 1

        else:
            cat = unicodedata.category(c)
            if cat in ('Mn', 'Me', 'Cf') and c != '\u00ad':
                current[-1][2] += c
                i += 1
            else:
                flush_sep()
                w = 2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1
                current.append([0, stack, c, w])
                i += 1

    if len(lines) <= max_lines:
        flush_sep()
        remaining = max_lines + 1 - len(lines)
        folded = fold(current, remaining)
        lines.extend(folded)

    truncated = len(lines) > max_lines
    if truncated:
        lines = lines[:max_lines]
        last = lines[-1]
        cw = sum(u[2] for u in last)
        limit = width - 1
        while last and cw > limit:
            cw -= last.pop()[2]
        last.append([None, '…', 1])

    result = []

    for units in lines:
        cw = sum(u[2] for u in units)
        p = width - cw

        if align == 'left':
            padded = units + [[None, ' ', 1] for _ in range(p)]
        elif align == 'right':
            padded = [[None, ' ', 1] for _ in range(p)] + units
        else:
            left = p // 2
            right = p - left
            padded = (
                [[None, ' ', 1] for _ in range(left)]
                + units
                + [[None, ' ', 1] for _ in range(right)]
            )

        runs = []
        for node, text, _ in padded:
            if runs and runs[-1][0] is node:
                runs[-1][1] += text
            else:
                runs.append([node, text])

        rendered = []
        for node, text in runs:
            if node is None:
                styles = []
            else:
                styles = []
                cur = node
                while cur is not None:
                    styles.append(cur[1])
                    cur = cur[0]
                styles.reverse()
            rendered.append([styles, text])

        result.append(rendered)

    return result