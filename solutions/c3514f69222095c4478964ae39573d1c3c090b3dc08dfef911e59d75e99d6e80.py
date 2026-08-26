def reconcile_bundle(directories, files, release_files, protected_directories, delete_delays, scan_round):
    import bisect

    def managed_path(p):
        return p.startswith("templates/") and p.rsplit("/", 1)[-1].endswith(".tpl")

    managed = {}
    all_files = []
    for row in files:
        p, c = row
        all_files.append(p)
        if managed_path(p):
            managed[p] = c

    unsafe = set()
    skipped = set()
    candidates = {}
    collisions = set()

    for raw in release_files:
        parts = raw.replace("\\", "/").split("/")
        parts = [x for x in parts if x and x != "."]
        if not parts or parts[0] != "bundle":
            skipped.add(raw)
            continue

        stack = []
        bad = False
        for x in parts[1:]:
            if x == "..":
                if not stack:
                    bad = True
                    break
                stack.pop()
            else:
                stack.append(x)

        if bad:
            unsafe.add(raw)
            continue

        dest = "/".join(stack)
        if not dest or raw.endswith("/") or raw.endswith("\\") or not managed_path(dest):
            skipped.add(raw)
            continue

        old = candidates.get(dest)
        if old is not None:
            collisions.add(dest)
            skipped.add(old)
            skipped.add(raw)
        else:
            candidates[dest] = raw

    archive_writes = sorted(p for p in candidates if p not in collisions)
    written = set(archive_writes)

    protected = sorted(set(protected_directories))
    protected_starts = sorted(p + "/" for p in protected)
    protected_ends = sorted(p + "0" for p in protected)

    def is_protected(path):
        return (
            bisect.bisect_right(protected_starts, path)
            > bisect.bisect_right(protected_ends, path)
        )

    orphan_attempts = []
    failed_deletions = []
    pending_visibility = []
    accepted = set()
    visible = []

    for p in all_files:
        if p not in managed:
            visible.append(p)
            continue

        if p in written or is_protected(p):
            visible.append(p)
            continue

        orphan_attempts.append(p)
        d = delete_delays.get(p, 0)

        if d == -1:
            failed_deletions.append(p)
            visible.append(p)
        else:
            accepted.add(p)
            if d > scan_round:
                pending_visibility.append(p)
                visible.append(p)

    index = [[p, managed[p]] for p in managed if p not in accepted]
    index.sort(key=lambda x: x[0])

    visible.sort()
    orphan_attempts.sort()
    failed_deletions.sort()
    pending_visibility.sort()

    items = sorted(set(visible) | set(protected))
    removed_directories = []

    for d in directories:
        if d == "templates" or not d.startswith("templates/"):
            continue
        if d in protected:
            continue
        prefix = d + "/"
        i = bisect.bisect_left(items, prefix)
        if i >= len(items) or not items[i].startswith(prefix):
            removed_directories.append(d)

    removed_directories.sort()

    return {
        "archive_writes": archive_writes,
        "unsafe_entries": sorted(unsafe),
        "skipped_entries": sorted(skipped),
        "collisions": sorted(collisions),
        "orphan_attempts": orphan_attempts,
        "failed_deletions": failed_deletions,
        "pending_visibility": pending_visibility,
        "visible_files": visible,
        "removed_directories": removed_directories,
        "index": index,
    }