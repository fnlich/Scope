use std::collections::BTreeMap;
use std::io::{self, Read, Write};

#[derive(Clone, Copy)]
struct Info {
    count: i32,
    cond: i32,
    xor: u64,
    max_depth: i32,
    max_node: usize,
}

impl Info {
    fn empty() -> Self {
        Self {
            count: 0,
            cond: 0,
            xor: 0,
            max_depth: -1,
            max_node: 0,
        }
    }

    fn merge(a: Self, b: Self) -> Self {
        let (max_depth, max_node) = if a.max_depth >= b.max_depth {
            (a.max_depth, a.max_node)
        } else {
            (b.max_depth, b.max_node)
        };
        Self {
            count: a.count + b.count,
            cond: a.cond + b.cond,
            xor: a.xor ^ b.xor,
            max_depth,
            max_node,
        }
    }
}

struct SegTree {
    size: usize,
    tree: Vec<Info>,
    depth: Vec<i32>,
    node_at_pos: Vec<usize>,
}

impl SegTree {
    fn new(n: usize, depth: Vec<i32>, node_at_pos: Vec<usize>) -> Self {
        let mut size = 1;
        while size < n {
            size <<= 1;
        }
        Self {
            size,
            tree: vec![Info::empty(); size << 1],
            depth,
            node_at_pos,
        }
    }

    fn point_add(&mut self, pos: usize, dc: i32, dcond: i32, id: u64) {
        let mut p = self.size + pos;
        self.tree[p].count += dc;
        self.tree[p].cond += dcond;
        self.tree[p].xor ^= id;
        if self.tree[p].count > 0 {
            self.tree[p].max_depth = self.depth[pos];
            self.tree[p].max_node = self.node_at_pos[pos];
        } else {
            self.tree[p].max_depth = -1;
            self.tree[p].max_node = 0;
        }
        p >>= 1;
        while p > 0 {
            self.tree[p] = Info::merge(self.tree[p << 1], self.tree[p << 1 | 1]);
            p >>= 1;
        }
    }

    fn range_query(&self, mut l: usize, mut r: usize) -> Info {
        l += self.size;
        r += self.size;
        let mut left = Info::empty();
        let mut right = Info::empty();
        while l < r {
            if l & 1 != 0 {
                left = Info::merge(left, self.tree[l]);
                l += 1;
            }
            if r & 1 != 0 {
                r -= 1;
                right = Info::merge(self.tree[r], right);
            }
            l >>= 1;
            r >>= 1;
        }
        Info::merge(left, right)
    }

    fn get(&self, pos: usize) -> Info {
        self.tree[self.size + pos]
    }
}

#[derive(Clone)]
struct Candidate {
    l: usize,
    r: usize,
    kind: u8,
    x: i64,
    cond: bool,
}

fn map_add(map: &mut BTreeMap<i64, (i32, u64)>, x: i64, id: u64) {
    let e = map.entry(x).or_insert((0, 0));
    e.0 += 1;
    e.1 ^= id;
}

fn map_remove(map: &mut BTreeMap<i64, (i32, u64)>, x: i64, id: u64) {
    let mut remove = false;
    if let Some(e) = map.get_mut(&x) {
        e.0 -= 1;
        e.1 ^= id;
        if e.0 == 0 {
            remove = true;
        }
    }
    if remove {
        map.remove(&x);
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let s: usize = it.next().unwrap().parse().unwrap();
    let n: usize = it.next().unwrap().parse().unwrap();
    let t: usize = it.next().unwrap().parse().unwrap();

    let mut parent = vec![usize::MAX; s];
    let mut children = vec![Vec::<usize>::new(); s];
    let mut roots = Vec::new();

    for v in 0..s {
        let p: usize = it.next().unwrap().parse().unwrap();
        if p == 0 {
            roots.push(v);
        } else {
            let pp = p - 1;
            parent[v] = pp;
            children[pp].push(v);
        }
    }

    let mut q = Vec::with_capacity(t);
    for _ in 0..t {
        let x: usize = it.next().unwrap().parse().unwrap();
        q.push(x - 1);
    }

    let mut candidates = Vec::with_capacity(n);
    let mut starts = vec![Vec::<usize>::new(); t + 1];
    let mut ends = vec![Vec::<usize>::new(); t + 1];

    for id in 0..n {
        let l: usize = it.next().unwrap().parse().unwrap();
        let r: usize = it.next().unwrap().parse().unwrap();
        let outcome = it.next().unwrap().as_bytes()[0];
        let kind_byte = it.next().unwrap().as_bytes()[0];
        let x: i64 = it.next().unwrap().parse().unwrap();

        let kind = match kind_byte {
            b'I' => 0,
            b'O' => 1,
            _ => 2,
        };
        let cond = outcome == b'C';

        candidates.push(Candidate {
            l,
            r,
            kind,
            x,
            cond,
        });

        if outcome != b'F' {
            starts[l].push(id);
            if r < t {
                ends[r + 1].push(id);
            }
        }
    }

    let mut order = Vec::with_capacity(s);
    let mut stack = roots.clone();
    while let Some(v) = stack.pop() {
        order.push(v);
        for &u in &children[v] {
            stack.push(u);
        }
    }

    let mut depth = vec![0i32; s];
    for &v in &order {
        for &u in &children[v] {
            depth[u] = depth[v] + 1;
        }
    }

    let mut size_sub = vec![1usize; s];
    let mut heavy = vec![usize::MAX; s];
    for &v in order.iter().rev() {
        let mut best = 0usize;
        let mut total = 1usize;
        for &u in &children[v] {
            total += size_sub[u];
            if size_sub[u] > best {
                best = size_sub[u];
                heavy[v] = u;
            }
        }
        size_sub[v] = total;
    }

    let mut head = vec![0usize; s];
    let mut pos = vec![0usize; s];
    let mut node_at_pos = vec![0usize; s];
    let mut cur = 0usize;
    let mut chain_stack = Vec::<(usize, usize)>::new();

    for &r in &roots {
        chain_stack.push((r, r));
    }

    while let Some((start, h)) = chain_stack.pop() {
        let mut v = start;
        loop {
            head[v] = h;
            pos[v] = cur;
            node_at_pos[cur] = v;
            cur += 1;

            for &u in &children[v] {
                if u != heavy[v] {
                    chain_stack.push((u, u));
                }
            }

            if heavy[v] == usize::MAX {
                break;
            }
            v = heavy[v];
        }
    }

    let mut seg = SegTree::new(s, depth, node_at_pos);

    let mut o_map = BTreeMap::<i64, (i32, u64)>::new();
    let mut u_map = BTreeMap::<i64, (i32, u64)>::new();

    let mut non_i_count = 0i32;
    let mut non_i_cond = 0i32;
    let mut non_i_xor = 0u64;

    let mut output = String::new();

    for time in 1..=t {
        for &id0 in &ends[time] {
            let c = &candidates[id0];
            let id = (id0 + 1) as u64;

            match c.kind {
                0 => {
                    let v = c.x as usize - 1;
                    seg.point_add(pos[v], -1, if c.cond { -1 } else { 0 }, id);
                }
                1 => {
                    map_remove(&mut o_map, c.x, id);
                    non_i_count -= 1;
                    non_i_xor ^= id;
                    if c.cond {
                        non_i_cond -= 1;
                    }
                }
                _ => {
                    map_remove(&mut u_map, c.x, id);
                    non_i_count -= 1;
                    non_i_xor ^= id;
                    if c.cond {
                        non_i_cond -= 1;
                    }
                }
            }
        }

        for &id0 in &starts[time] {
            let c = &candidates[id0];
            let id = (id0 + 1) as u64;

            match c.kind {
                0 => {
                    let v = c.x as usize - 1;
                    seg.point_add(pos[v], 1, if c.cond { 1 } else { 0 }, id);
                }
                1 => {
                    map_add(&mut o_map, c.x, id);
                    non_i_count += 1;
                    non_i_xor ^= id;
                    if c.cond {
                        non_i_cond += 1;
                    }
                }
                _ => {
                    map_add(&mut u_map, c.x, id);
                    non_i_count += 1;
                    non_i_xor ^= id;
                    if c.cond {
                        non_i_cond += 1;
                    }
                }
            }
        }

        let target = q[time - 1];
        let mut path = Info::empty();
        let mut v = target;

        loop {
            let h = head[v];
            path = Info::merge(path, seg.range_query(pos[h], pos[v] + 1));
            if parent[h] == usize::MAX {
                break;
            }
            v = parent[h];
        }

        let total = non_i_count + path.count;
        let conditional = non_i_cond + path.cond;

        if total == 0 {
            output.push_str("FAILURE\n");
            continue;
        }

        if total == 1 {
            let id = non_i_xor ^ path.xor;
            output.push_str("SELECT ");
            output.push_str(&id.to_string());
            output.push('\n');
            continue;
        }

        if conditional > 0 {
            output.push_str("AMBIGUOUS\n");
            continue;
        }

        if path.count > 0 {
            let winner_node = path.max_node;
            let winner = seg.get(pos[winner_node]);
            if winner.count == 1 {
                output.push_str("SELECT ");
                output.push_str(&winner.xor.to_string());
                output.push('\n');
            } else {
                output.push_str("AMBIGUOUS\n");
            }
            continue;
        }

        if let Some((&rank, &(count, ids))) = o_map.iter().next() {
            let _ = rank;
            if count == 1 {
                output.push_str("SELECT ");
                output.push_str(&ids.to_string());
                output.push('\n');
            } else {
                output.push_str("AMBIGUOUS\n");
            }
            continue;
        }

        if let Some((&distance, &(count, ids))) = u_map.iter().next() {
            let _ = distance;
            if count == 1 {
                output.push_str("SELECT ");
                output.push_str(&ids.to_string());
                output.push('\n');
            } else {
                output.push_str("AMBIGUOUS\n");
            }
        } else {
            output.push_str("FAILURE\n");
        }
    }

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    out.write_all(output.as_bytes()).unwrap();
}