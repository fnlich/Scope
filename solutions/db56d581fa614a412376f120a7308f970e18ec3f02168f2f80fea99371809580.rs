use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::{self, Read, Write};

struct Scanner {
    data: Vec<u8>,
    pos: usize,
}

impl Scanner {
    fn new() -> Self {
        let mut data = Vec::new();
        io::stdin().read_to_end(&mut data).unwrap();
        Self { data, pos: 0 }
    }

    fn next_u64(&mut self) -> Option<u64> {
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return None;
        }
        let mut x = 0u64;
        while self.pos < self.data.len() && !self.data[self.pos].is_ascii_whitespace() {
            x = x * 10 + (self.data[self.pos] - b'0') as u64;
            self.pos += 1;
        }
        Some(x)
    }
}

struct Edge {
    v: usize,
    w: u64,
    c: usize,
    a: u64,
    b: u64,
}

struct Query {
    s: usize,
    t: usize,
    mask: u64,
    d: u64,
}

struct Answer {
    cost: u64,
    edges: Vec<u32>,
    a: u64,
    b: u64,
}

fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn main() {
    let mut sc = Scanner::new();

    let first = match sc.next_u64() {
        Some(x) => x,
        None => return,
    };

    let second = match sc.next_u64() {
        Some(x) => x,
        None => {
            let mut out = io::BufWriter::new(io::stdout().lock());
            writeln!(out, "{}", first).unwrap();
            return;
        }
    };

    let third = match sc.next_u64() {
        Some(x) => x,
        None => {
            let mut out = io::BufWriter::new(io::stdout().lock());
            writeln!(out, "{}", second).unwrap();
            return;
        }
    };

    let fourth = match sc.next_u64() {
        Some(x) => x,
        None => {
            let mut out = io::BufWriter::new(io::stdout().lock());
            writeln!(out, "{}", third).unwrap();
            return;
        }
    };

    let fifth = match sc.next_u64() {
        Some(x) => x,
        None => {
            let mut out = io::BufWriter::new(io::stdout().lock());
            writeln!(out, "{}", fourth).unwrap();
            return;
        }
    };

    let n = first as usize;
    let m = second as usize;
    let k = third as usize;
    let q = fourth as usize;
    let p = fifth;

    let mut graph = vec![Vec::<usize>::new(); n];
    let mut edges = Vec::<Edge>::with_capacity(m);

    for _ in 0..m {
        let u = sc.next_u64().unwrap() as usize - 1;
        let v = sc.next_u64().unwrap() as usize - 1;
        let w = sc.next_u64().unwrap();
        let c = sc.next_u64().unwrap() as usize;
        let a = sc.next_u64().unwrap();
        let b = sc.next_u64().unwrap();
        let id = edges.len();
        edges.push(Edge { v, w, c, a, b });
        graph[u].push(id);
    }

    let mut queries = Vec::with_capacity(q);
    for _ in 0..q {
        let s = sc.next_u64().unwrap() as usize - 1;
        let t = sc.next_u64().unwrap() as usize - 1;
        let mask = sc.next_u64().unwrap();
        let d = sc.next_u64().unwrap();
        queries.push(Query { s, t, mask, d });
    }

    let mut answers: Vec<Option<Answer>> = (0..q).map(|_| None).collect();
    let mut groups: HashMap<(usize, u64, u64), Vec<usize>> = HashMap::new();

    for (qi, query) in queries.iter().enumerate() {
        if query.s == query.t {
            let mut best_id = usize::MAX;
            let mut best_cost = u64::MAX;

            for &eid in &graph[query.s] {
                let e = &edges[eid];
                if e.v != query.s {
                    continue;
                }
                let enabled = ((query.mask >> e.c) & 1) != 0;
                let cost = if enabled {
                    e.w
                } else {
                    e.w + query.d
                };
                if cost < best_cost || (cost == best_cost && eid < best_id) {
                    best_cost = cost;
                    best_id = eid;
                }
            }

            if best_id == usize::MAX {
                answers[qi] = Some(Answer {
                    cost: 0,
                    edges: Vec::new(),
                    a: 1 % p,
                    b: 0,
                });
            } else {
                let e = &edges[best_id];
                answers[qi] = Some(Answer {
                    cost: best_cost,
                    edges: vec![best_id as u32],
                    a: e.a,
                    b: e.b,
                });
            }
        } else {
            groups
                .entry((query.s, query.mask, query.d))
                .or_default()
                .push(qi);
        }
    }

    let state_count = n * 2;
    let inf = u64::MAX;
    let invalid = usize::MAX;

    for ((s, mask, d), qs) in groups {
        let mut need = vec![false; n];
        let mut remaining = 0usize;

        for &qi in &qs {
            let t = queries[qi].t;
            if !need[t] {
                need[t] = true;
                remaining += 1;
            }
        }

        let mut dist = vec![inf; state_count];
        let mut rank = vec![invalid; state_count];
        let mut pred_state = vec![invalid; state_count];
        let mut pred_edge = vec![invalid; state_count];
        let mut done = vec![false; state_count];
        let mut target_state = vec![invalid; n];

        let source_state = s * 2;
        dist[source_state] = 0;
        rank[source_state] = 0;

        let mut next_rank = 1usize;
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((0u64, 0usize, source_state)));

        while let Some(Reverse((cost, arank, state))) = heap.pop() {
            if done[state] || dist[state] != cost || rank[state] != arank {
                continue;
            }

            done[state] = true;

            let frame = state / 2;
            if need[frame] && target_state[frame] == invalid {
                target_state[frame] = state;
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }

            let used = state & 1;

            for &eid in &graph[frame] {
                let e = &edges[eid];
                let enabled = ((mask >> e.c) & 1) != 0;

                if !enabled && used == 1 {
                    continue;
                }

                let next_used = if enabled { used } else { 1 };
                let next_state = e.v * 2 + next_used;

                if done[next_state] {
                    continue;
                }

                let add = if enabled { e.w } else { e.w + d };
                let new_cost = cost + add;

                if new_cost < dist[next_state] {
                    dist[next_state] = new_cost;
                    rank[next_state] = next_rank;
                    pred_state[next_state] = state;
                    pred_edge[next_state] = eid;
                    heap.push(Reverse((new_cost, next_rank, next_state)));
                    next_rank += 1;
                }
            }
        }

        for &qi in &qs {
            let t = queries[qi].t;
            let state = target_state[t];

            if state == invalid {
                continue;
            }

            let mut route = Vec::<u32>::new();
            let mut cur = state;

            while cur != source_state {
                let eid = pred_edge[cur];
                if eid == invalid {
                    route.clear();
                    break;
                }
                route.push(eid as u32);
                cur = pred_state[cur];
            }

            if cur != source_state {
                continue;
            }

            route.reverse();

            let mut a_acc = 1 % p;
            let mut b_acc = 0u64;

            for &eid32 in &route {
                let e = &edges[eid32 as usize];
                a_acc = mul_mod(e.a, a_acc, p);
                b_acc = ((mul_mod(e.a, b_acc, p) as u128 + e.b as u128) % p as u128) as u64;
            }

            answers[qi] = Some(Answer {
                cost: dist[state],
                edges: route,
                a: a_acc,
                b: b_acc,
            });
        }
    }

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for answer in answers {
        match answer {
            None => {
                writeln!(out, "NONE").unwrap();
            }
            Some(ans) => {
                write!(out, "{} {} {} {}", ans.cost, ans.edges.len(), ans.a, ans.b).unwrap();
                for eid in ans.edges {
                    write!(out, " {}", eid + 1).unwrap();
                }
                writeln!(out).unwrap();
            }
        }
    }
}