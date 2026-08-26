def decode_checkpoints(initial_locals, code_size, max_locals, max_stack, allocations, initializations, records):
    def slots(seq):
        total = 0
        for x in seq:
            total += 2 if x == "J" or x == "D" else 1
        return total

    def resolve(seq, offset):
        out = []
        for x in seq:
            if isinstance(x, str) and x.startswith("U:"):
                label = x[2:]
                init = initializations.get(label)
                if init is not None and init[0] < offset:
                    out.append(init[1])
                else:
                    out.append(x)
            else:
                out.append(x)
        return tuple(out)

    def live(x, offset):
        if not (isinstance(x, str) and x.startswith("U:")):
            return True
        label = x[2:]
        a = allocations[label]
        init = initializations.get(label)
        return a < offset and (init is None or init[0] >= offset)

    def validate_new(seq, offset):
        for x in seq:
            if isinstance(x, str) and x.startswith("U:") and not live(x, offset):
                return False
        return True

    frames = []
    prev_offset = None
    prev_locals = tuple(initial_locals)
    first = True
    full_mode = None

    for i, rec in enumerate(records):
        kind = rec[0]

        if kind == "append" or kind == "chop":
            count = rec[2]
            if count < 1 or count > 3:
                return ("ERROR", i, "COUNT")

        delta = rec[1]
        if delta < 0:
            return ("ERROR", i, "DELTA")

        if first:
            offset = delta
        else:
            offset = prev_offset + delta + 1

        if offset < 0 or offset >= code_size:
            return ("ERROR", i, "LOCATION")

        is_full = kind == "full"
        if first:
            full_mode = is_full
        elif is_full != full_mode:
            return ("ERROR", i, "MIXED")

        if is_full:
            locals_seq = tuple(rec[2])
            stack_seq = tuple(rec[3])
        else:
            locals_seq = prev_locals
            stack_seq = ()

            if kind == "same":
                pass
            elif kind == "one":
                stack_seq = (rec[2],)
            elif kind == "append":
                locals_seq = locals_seq + tuple(rec[2])
            elif kind == "chop":
                count = rec[2]
                if count > len(locals_seq):
                    return ("ERROR", i, "UNDERFLOW")
                locals_seq = locals_seq[:-count]

        if is_full:
            scan_sequences = (locals_seq, stack_seq)
        elif kind == "one":
            scan_sequences = (stack_seq,)
        elif kind == "append":
            scan_sequences = (tuple(rec[2]),)
        else:
            scan_sequences = ()

        token_ok = True
        for seq in scan_sequences:
            if not validate_new(seq, offset):
                token_ok = False
                break
        if not token_ok:
            return ("ERROR", i, "TOKEN")

        if locals_seq and locals_seq[-1] == "TOP":
            return ("ERROR", i, "LOCALS")
        if slots(locals_seq) > max_locals:
            return ("ERROR", i, "LOCALS")
        if slots(stack_seq) > max_stack:
            return ("ERROR", i, "STACK")

        frames.append((offset, resolve(locals_seq, offset), tuple(stack_seq)))
        prev_locals = locals_seq
        prev_offset = offset
        first = False

    return ("OK", frames)