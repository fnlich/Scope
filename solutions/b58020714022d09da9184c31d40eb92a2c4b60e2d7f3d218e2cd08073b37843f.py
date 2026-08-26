import bisect

def segmented_top(operations):
    class BIT:
        __slots__ = ('n', 't')
        def __init__(self, n):
            self.n = n
            self.t = [0]*(n+1)
        def update(self, i, delta):
            i += 1
            while i <= self.n:
                self.t[i] += delta
                i += i & (-i)
        def prefix(self, i):
            i += 1
            s = 0
            while i > 0:
                s += self.t[i]
                i -= i & (-i)
            return s
        def find_kth(self, k):
            pos = 0
            rem = k
            LOG = self.n.bit_length()
            for pw in range(LOG, -1, -1):
                np = pos + (1 << pw)
                if np <= self.n and self.t[np] < rem:
                    pos = np
                    rem -= self.t[np]
            return pos

    doc_stack = {}
    doc_current = {}
    seg_docs = {}
    seg_active = {}
    seg_dict = {}

    all_keys = {}
    for name in ('asc_first','asc_last','desc_first','desc_last'):
        all_keys[name] = []

    entries = []
    n_ops = len(operations)

    add_time_counter = [0]

    def make_val(tag, s):
        if tag == 's':
            return ('string', s)
        else:
            return ('null', None)

    def sort_key_str(direction, s):
        if direction == 'asc':
            return s
        else:
            return None

    class ReverseStr(str):
        def __lt__(self, other):
            return str.__gt__(self, other)
        def __gt__(self, other):
            return str.__lt__(self, other)
        def __le__(self, other):
            return str.__ge__(self, other)
        def __ge__(self, other):
            return str.__le__(self, other)

    configs = [('asc','first'), ('asc','last'), ('desc','first'), ('desc','last')]

    doc_key_for_config = {c: {} for c in configs}
    doc_present = {}
    doc_quality = {}
    doc_val = {}
    doc_seg_time = {}

    key_lists = {c: [] for c in configs}

    ops_processed = []
    live_docs = set()

    for op in operations:
        if op[0] == 'add':
            _, seg_id, dictionary, rows = op
            seg_dict[seg_id] = dictionary
            seg_active[seg_id] = True
            add_time_counter[0] += 1
            t = add_time_counter[0]
            doc_ids_here = []
            for (doc_id, tag, payload, quality) in rows:
                s = None
                if tag == 's':
                    s = dictionary[payload]
                if doc_id not in doc_stack:
                    doc_stack[doc_id] = []
                doc_stack[doc_id].append((t, seg_id, tag, s, quality))
                doc_ids_here.append(doc_id)
            seg_docs[seg_id] = doc_ids_here
            for c in configs:
                for k in list(key_lists[c]):
                    pass
        elif op[0] == 'drop':
            _, seg_id = op
            seg_active[seg_id] = False
        else:
            pass
        ops_processed.append(op)

    all_doc_ids = set()
    for d in doc_stack:
        all_doc_ids.add(d)
    doc_id_list = sorted(all_doc_ids)
    doc_idx = {d:i for i,d in enumerate(doc_id_list)}
    n_docs = len(doc_id_list)

    def strsign_asc(s):
        return s
    class RevStr:
        __slots__=('s',)
        def __init__(self, s):
            self.s = s
        def __lt__(self, other):
            return self.s > other.s
        def __eq__(self, other):
            return self.s == other.s

    def current_key(tag, s, direction, nulls):
        if tag == 'n':
            if nulls == 'first':
                return (0,)
            else:
                return (1,)
        else:
            if nulls == 'first':
                grp = 1
            else:
                grp = 0
            if direction == 'asc':
                return (grp, 0, s)
            else:
                return (grp, 1, RevStr(s))

    seg_active2 = {}
    doc_stack2 = {}
    for d, st in doc_stack.items():
        doc_stack2[d] = st

    seg_active_map = {}
    for op in operations:
        if op[0] == 'add':
            seg_active_map[op[1]] = True

    active = dict(seg_active_map)

    def top_occurrence(doc_id):
        st = doc_stack2.get(doc_id, [])
        best = None
        for entry in st:
            t, seg_id, tag, s, quality = entry
            if active.get(seg_id, False):
                if best is None or entry[0] > best[0]:
                    best = entry
        return best

    fen_configs = {}
    for c in configs:
        fen_configs[c] = None

    class KeySet:
        def __init__(self):
            self.keys = {}
        def build(self, sample_keys):
            uniq = sorted(set(sample_keys))
            self.rank = {k:i for i,k in enumerate(uniq)}
            self.size = len(uniq)
            self.bit = BIT(self.size)
        def idx(self, key):
            return self.rank[key]

    all_possible_keys = {c: [] for c in configs}
    for doc_id, st in doc_stack2.items():
        for (t, seg_id, tag, s, quality) in st:
            for direction, nulls in configs:
                k = current_key(tag, s, direction, nulls)
                all_possible_keys[(direction,nulls)].append(k)

    keyset = {}
    for c in configs:
        ks = KeySet()
        ks.build(all_possible_keys[c])
        keyset[c] = ks

    doc_meta = {}
    for doc_id in doc_stack2:
        doc_meta[doc_id] = None

    present_state = {}

    def compute_present(doc_id):
        occ = top_occurrence(doc_id)
        if occ is None:
            return None
        t, seg_id, tag, s, quality = occ
        if tag == 'x':
            return None
        return (tag, s, quality, doc_id)

    results = []

    active_seg = {}

    def process_add(seg_id, dictionary, rows):
        active_seg[seg_id] = True
        add_time_counter[0] += 1
        t = add_time_counter[0]
        for (doc_id, tag, payload, quality) in rows:
            s = None
            if tag == 's':
                s = dictionary[payload]
            doc_stack2.setdefault(doc_id, []).append((t, seg_id, tag, s, quality))
            update_doc(doc_id)

    def process_drop(seg_id):
        active_seg[seg_id] = False
        for doc_id in seg_docs.get(seg_id, []):
            update_doc(doc_id)

    def remove_from_structures(doc_id):
        old = present_state.get(doc_id)
        if old is not None:
            tag, s, quality, did = old
            for c in configs:
                key = current_key(tag, s, c[0], c[1])
                idx = keyset[c].idx(key)
                keyset[c].bit.update(idx, -1)
                comp_lists[c][idx].discard((quality, did))
            present_state[doc_id] = None

    def add_to_structures(doc_id, tag, s, quality):
        for c in configs:
            key = current_key(tag, s, c[0], c[1])
            idx = keyset[c].idx(key)
            keyset[c].bit.update(idx, 1)
            comp_lists[c][idx].add((quality, doc_id))
        present_state[doc_id] = (tag, s, quality, doc_id)

    comp_lists = {}
    for c in configs:
        comp_lists[c] = [set() for _ in range(keyset[c].size)]

    def update_doc(doc_id):
        occ = top_occurrence(doc_id)
        remove_from_structures(doc_id)
        if occ is None:
            return
        t, seg_id, tag, s, quality = occ
        if tag == 'x':
            return
        add_to_structures(doc_id, tag, s, quality)

    for op in operations:
        if op[0] == 'add':
            _, seg_id, dictionary, rows = op
            process_add(seg_id, dictionary, rows)
        elif op[0] == 'drop':
            _, seg_id = op
            process_drop(seg_id)
        else:
            _, k, direction, nulls, min_quality = op
            c = (direction, nulls)
            ks = keyset[c]
            bit = ks.bit
            comp = comp_lists[c]
            total = bit.prefix(ks.size-1) if ks.size>0 else 0
            answer = []
            if k > 0 and ks.size > 0:
                remaining = k
                idx = 0
                cum = 0
                sizearr = ks.size
                bititems = bit
                pos = 0
                while remaining > 0 and pos < sizearr:
                    cnt = bititems.t
                    npos = pos
                    step = 1 << (sizearr.bit_length())
                    break
                for idx in range(sizearr):
                    if remaining <= 0:
                        break
                    cval = bit.prefix(idx) - (bit.prefix(idx-1) if idx>0 else 0)
                    if cval == 0:
                        continue
                    bucket = comp[idx]
                    if not bucket:
                        continue
                    cand = sorted(bucket, key=lambda x: (-x[0], x[1]))
                    for (q, did) in cand:
                        if q < min_quality:
                            continue
                        st = doc_stack2[did]
                        occ = None
                        for e in st:
                            if e[1]==None:
                                pass
                        occ_info = present_state[did]
                        tag = occ_info[0]
                        s = occ_info[1]
                        val = make_val(tag, s)
                        answer.append((did, val, q))
                        remaining -= 1
                        if remaining == 0:
                            break
            results.append(answer)

    return results