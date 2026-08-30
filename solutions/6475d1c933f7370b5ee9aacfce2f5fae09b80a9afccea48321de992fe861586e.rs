use std::io::{self, Read, Write};
use std::collections::{HashMap, BTreeSet};

fn push_u64(out: &mut Vec<u8>, mut v: u64) {
    let mut buf = [0u8; 20];
    let mut i = 20;
    if v == 0 { i -= 1; buf[i] = b'0'; }
    while v > 0 { i -= 1; buf[i] = b'0' + (v % 10) as u8; v /= 10; }
    out.extend_from_slice(&buf[i..]);
}

struct Scanner { data: Vec<u8>, p: usize }
impl Scanner {
    fn new(d: Vec<u8>) -> Self { Scanner { data: d, p: 0 } }
    fn tok(&mut self) -> &[u8] {
        while self.p < self.data.len() && self.data[self.p].is_ascii_whitespace() { self.p += 1; }
        let s = self.p;
        while self.p < self.data.len() && !self.data[self.p].is_ascii_whitespace() { self.p += 1; }
        &self.data[s..self.p]
    }
    fn u64(&mut self) -> u64 {
        let t = self.tok();
        let mut v: u64 = 0;
        for &c in t { v = v * 10 + (c - b'0') as u64; }
        v
    }
    fn i64(&mut self) -> i64 {
        let t = self.tok();
        let mut neg = false;
        let mut v: i64 = 0;
        for &c in t {
            if c == b'-' { neg = true; } else if c == b'+' { } else { v = v * 10 + (c - b'0') as i64; }
        }
        if neg { -v } else { v }
    }
    fn ch(&mut self) -> u8 {
        let t = self.tok();
        if t.is_empty() { b'?' } else { t[0] }
    }
}

fn main() {
    let mut inp = Vec::new();
    io::stdin().read_to_end(&mut inp).unwrap();
    let mut sc = Scanner::new(inp);
    let n = sc.u64() as usize;
    let q = sc.u64() as usize;

    let mut id = vec![0u64; n];
    let mut kind = vec![0u8; n];
    let mut mode = vec![0u8; n];
    let mut prio = vec![0i64; n];
    let mut mirror = vec![0u64; n];
    let mut enabled = vec![false; n];
    let mut idx: HashMap<u64, usize> = HashMap::with_capacity(n * 2);

    for i in 0..n {
        id[i] = sc.u64();
        kind[i] = sc.ch();
        mode[i] = sc.ch();
        prio[i] = sc.i64();
        mirror[i] = sc.u64();
        enabled[i] = sc.u64() == 1;
        idx.insert(id[i], i);
    }

    let mut state = vec![0u8; n];
    let mut rt = vec![0u8; n];
    let mut rv = vec![0u64; n];
    let mut path: Vec<usize> = Vec::new();

    for i in 0..n {
        if state[i] != 0 { continue; }
        path.clear();
        let mut cur = i;
        let ft: u8;
        let fv: u64;
        loop {
            state[cur] = 1;
            path.push(cur);
            let m = mirror[cur];
            if m == 0 { ft = 0; fv = id[cur]; break; }
            match idx.get(&m) {
                None => { ft = 1; fv = m; break; }
                Some(&j) => {
                    if state[j] == 0 { cur = j; continue; }
                    else if state[j] == 2 { ft = rt[j]; fv = rv[j]; break; }
                    else {
                        let mut mn = id[j];
                        let mut k = j;
                        loop {
                            let nx = *idx.get(&mirror[k]).unwrap();
                            if nx == j { break; }
                            k = nx;
                            if id[k] < mn { mn = id[k]; }
                        }
                        ft = 2; fv = mn; break;
                    }
                }
            }
        }
        for &p in path.iter() { state[p] = 2; rt[p] = ft; rv[p] = fv; }
    }

    let mut out: Vec<u8> = Vec::with_capacity(1 << 22);
    for i in 0..n {
        match rt[i] {
            0 => out.extend_from_slice(b"OK "),
            1 => out.extend_from_slice(b"BROKEN "),
            _ => out.extend_from_slice(b"CYCLE "),
        }
        push_u64(&mut out, rv[i]);
        out.push(b'\n');
    }

    let mut child_head = vec![usize::MAX; n];
    let mut children: Vec<Vec<usize>> = Vec::new();
    let mut childlist: Vec<Vec<usize>> = vec![Vec::new(); n];
    let _ = &mut children;
    let _ = &mut child_head;

    for i in 0..n {
        if rt[i] == 0 && mirror[i] != 0 {
            let p = *idx.get(&mirror[i]).unwrap();
            childlist[p].push(i);
        }
    }
    for i in 0..n {
        if childlist[i].len() > 1 {
            let pr = &prio;
            let ids = &id;
            childlist[i].sort_by(|&a, &b| {
                (pr[a], ids[a]).cmp(&(pr[b], ids[b]))
            });
        }
    }

    let mut pos = vec![usize::MAX; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        if rt[i] == 0 && mirror[i] == 0 {
            stack.clear();
            stack.push(i);
            while let Some(v) = stack.pop() {
                pos[v] = order.len();
                order.push(v);
                for k in (0..childlist[v].len()).rev() {
                    stack.push(childlist[v][k]);
                }
            }
        }
    }

    let mut sz = vec![1usize; n];
    for k in (0..order.len()).rev() {
        let v = order[k];
        if mirror[v] != 0 {
            let p = *idx.get(&mirror[v]).unwrap();
            sz[p] += sz[v];
        }
    }

    let mut root = vec![usize::MAX; n];
    for i in 0..n {
        if rt[i] == 0 {
            root[i] = *idx.get(&rv[i]).unwrap();
        }
    }

    let mut a_set: BTreeSet<usize> = BTreeSet::new();
    let mut h_set: BTreeSet<usize> = BTreeSet::new();
    for i in 0..n {
        if rt[i] == 0 && kind[i] == b'P' && enabled[i] {
            a_set.insert(pos[i]);
            if mode[i] == b'H' || mode[i] == b'B' { h_set.insert(pos[i]); }
        }
    }

    for _ in 0..q {
        let op = sc.ch();
        if op == b'E' {
            let t = sc.u64();
            let e = sc.u64() == 1;
            match idx.get(&t) {
                None => { out.extend_from_slice(b"UNKNOWN\n"); }
                Some(&i) => {
                    enabled[i] = e;
                    if rt[i] == 0 && kind[i] == b'P' {
                        let p = pos[i];
                        if e {
                            a_set.insert(p);
                            if mode[i] == b'H' || mode[i] == b'B' { h_set.insert(p); } else { h_set.remove(&p); }
                        } else {
                            a_set.remove(&p);
                            h_set.remove(&p);
                        }
                    }
                    out.extend_from_slice(b"UPDATED\n");
                }
            }
        } else {
            let t = sc.u64();
            let act = sc.ch();
            match idx.get(&t) {
                None => { out.extend_from_slice(b"UNKNOWN\n"); }
                Some(&i) => {
                    if rt[i] == 1 {
                        out.extend_from_slice(b"BROKEN ");
                        push_u64(&mut out, rv[i]);
                        out.push(b'\n');
                    } else if rt[i] == 2 {
                        out.extend_from_slice(b"CYCLE ");
                        push_u64(&mut out, rv[i]);
                        out.push(b'\n');
                    } else {
                        let r = root[i];
                        let lo = pos[r];
                        let hi = lo + sz[r];
                        let mut res: Vec<u64> = Vec::new();
                        if act == b'H' {
                            for &p in h_set.range(lo..hi) {
                                res.push(id[order[p]]);
                            }
                        } else {
                            let mut cur = lo;
                            loop {
                                let nx = match a_set.range(cur..hi).next() { Some(&x) => x, None => break };
                                let v = order[nx];
                                if mode[v] == b'S' || mode[v] == b'B' { res.push(id[v]); }
                                cur = nx + sz[v];
                                if cur >= hi { break; }
                            }
                        }
                        push_u64(&mut out, res.len() as u64);
                        for x in res {
                            out.push(b' ');
                            push_u64(&mut out, x);
                        }
                        out.push(b'\n');
                    }
                }
            }
        }
    }

    let so = io::stdout();
    let mut w = io::BufWriter::new(so.lock());
    w.write_all(&out).unwrap();
}