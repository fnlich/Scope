def settle_filings(initial, requests):
    def safe_tree(tree):
        out = {}
        for rec in tree:
            if not isinstance(rec, (list, tuple)) or len(rec) != 3:
                return None
            p, k, v = rec
            if not isinstance(p, str) or not isinstance(k, str) or not isinstance(v, str):
                return None
            if not p or p[0] == "/" or p[-1] == "/":
                return None
            parts = p.split("/")
            if any(x == "" or x == "." or x == ".." for x in parts):
                return None
            if p in out:
                return None
            if k == "D":
                if v != "":
                    return None
            elif k != "F":
                return None
            out[p] = (k, v)
        for p, (k, _) in out.items():
            if "/" in p:
                parent = p.rsplit("/", 1)[0]
                x = out.get(parent)
                if x is None or x[0] != "D":
                    return None
        x = out.get("manifest.rec")
        if x is None or x[0] != "F":
            return None
        return out

    def state_from_tree(tree):
        s = safe_tree(tree)
        return (tree, s)

    vault = {}
    for account, tree in initial:
        if not isinstance(account, str) or not account or account in vault:
            return ["INVALID"] * len(requests)
        vault[account] = state_from_tree(tree)

    ans = []

    for req in requests:
        account, base, edits, race = req

        if not isinstance(account, str) or not account:
            ans.append("INVALID")
            continue
        if base is not None and (not isinstance(base, str) or not base):
            ans.append("INVALID")
            continue
        if not isinstance(edits, list):
            ans.append("INVALID")
            continue

        if base is None:
            source = {}
        else:
            bs = vault.get(base)
            if bs is None or bs[1] is None:
                ans.append("INVALID")
                continue
            source = bs[1].copy()

        valid = True
        removals = set()
        additions = {}
        r_count = set()
        n_count = set()

        for e in edits:
            if not isinstance(e, (list, tuple)) or len(e) != 3:
                valid = False
                break
            p, k, v = e
            if not isinstance(p, str) or not isinstance(k, str) or not isinstance(v, str):
                valid = False
                break
            if not p or p[0] == "/" or p[-1] == "/":
                valid = False
                break
            parts = p.split("/")
            if any(x == "" or x == "." or x == ".." for x in parts):
                valid = False
                break

            if k == "R":
                if v != "" or p in r_count:
                    valid = False
                    break
                r_count.add(p)
                removals.add(p)
            else:
                if p in n_count:
                    valid = False
                    break
                n_count.add(p)
                if k not in ("D", "F"):
                    valid = False
                    break
                if k == "D" and v != "":
                    valid = False
                    break
                additions[p] = (k, v)

        if not valid:
            ans.append("INVALID")
            continue

        for p in removals:
            if p in source:
                del source[p]
            prefix = p + "/"
            for q in list(source):
                if q.startswith(prefix):
                    del source[q]

        for p, kv in additions.items():
            source[p] = kv

        good = True
        for p, (k, v) in source.items():
            if k not in ("D", "F"):
                good = False
                break
            if k == "D" and v != "":
                good = False
                break
            if "/" in p:
                parent = p.rsplit("/", 1)[0]
                x = source.get(parent)
                if x is None or x[0] != "D":
                    good = False
                    break
        if good:
            x = source.get("manifest.rec")
            if x is None or x[0] != "F":
                good = False

        if not good:
            ans.append("INVALID")
            continue

        existing = vault.get(account)
        if existing is not None:
            old = existing[1]
            if old is not None and old == source:
                ans.append("ALREADY_IDENTICAL")
            else:
                ans.append("CONFLICTING")
            continue

        if race is not None:
            vault[account] = state_from_tree(race)
            ans.append("RACE_ABORTED")
        else:
            vault[account] = (None, source)
            ans.append("INSTALLED")

    return ans