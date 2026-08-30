use std::io::{self, Read, Write};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
struct FastHash(u64);
impl Hasher for FastHash {
    fn finish(&self) -> u64 {
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0 ^ i).wrapping_mul(0x9E3779B97F4A7C15);
    }
}

type Map = HashMap<u64, u32, BuildHasherDefault<FastHash>>;

fn key(node: u32, ch: u32) -> u64 {
    ((node as u64) << 32) | (ch as u64)
}

fn main() {
    let mut inp = Vec::new();
    io::stdin().read_to_end(&mut inp).unwrap();
    let mut pos = 0usize;
    let mut next = || -> u64 {
        while pos < inp.len() && !(inp[pos] >= b'0' && inp[pos] <= b'9') {
            pos += 1;
        }
        let mut v: u64 = 0;
        while pos < inp.len() && inp[pos] >= b'0' && inp[pos] <= b'9' {
            v = v * 10 + (inp[pos] - b'0') as u64;
            pos += 1;
        }
        v
    };

    let k = next() as usize;

    let mut edges: Map = Map::default();
    let mut link: Vec<u32> = vec![0];
    let mut children: Vec<Vec<(u32, u32)>> = vec![Vec::new()];
    let mut own_len: Vec<u32> = vec![0];
    let mut own_id: Vec<u32> = vec![0];

    let mut insert = |seq: &Vec<u32>,
                      id: u32,
                      edges: &mut Map,
                      link: &mut Vec<u32>,
                      children: &mut Vec<Vec<(u32, u32)>>,
                      own_len: &mut Vec<u32>,
                      own_id: &mut Vec<u32>| {
        let mut cur: u32 = 0;
        for idx in (0..seq.len()).rev() {
            let ch = seq[idx];
            let kk = key(cur, ch);
            let nx = match edges.get(&kk) {
                Some(&w) => w,
                None => {
                    let w = link.len() as u32;
                    link.push(0);
                    children.push(Vec::new());
                    own_len.push(0);
                    own_id.push(0);
                    edges.insert(kk, w);
                    children[cur as usize].push((ch, w));
                    w
                }
            };
            cur = nx;
        }
        own_len[cur as usize] = seq.len() as u32;
        own_id[cur as usize] = id;
    };

    for t in 1..=k {
        let a = next() as usize;
        let mut open = Vec::with_capacity(a);
        for _ in 0..a {
            open.push(next() as u32);
        }
        let b = next() as usize;
        let mut close = Vec::with_capacity(b);
        for _ in 0..b {
            close.push(next() as u32);
        }
        let base = ((t - 1) as u32) * 2;
        insert(&open, base + 1, &mut edges, &mut link, &mut children, &mut own_len, &mut own_id);
        insert(&close, base + 2, &mut edges, &mut link, &mut children, &mut own_len, &mut own_id);
    }

    let nn = link.len();
    let mut best_len: Vec<u32> = vec![0; nn];
    let mut best_id: Vec<u32> = vec![0; nn];

    let mut queue: Vec<u32> = Vec::with_capacity(nn);
    {
        let rc = children[0].clone();
        for &(_, v) in rc.iter() {
            link[v as usize] = 0;
            queue.push(v);
        }
    }
    let mut qi = 0usize;
    while qi < queue.len() {
        let u = queue[qi];
        qi += 1;
        let lu = link[u as usize];
        if own_len[u as usize] >= best_len[lu as usize] && own_len[u as usize] > 0 {
            best_len[u as usize] = own_len[u as usize];
            best_id[u as usize] = own_id[u as usize];
        } else {
            best_len[u as usize] = best_len[lu as usize];
            best_id[u as usize] = best_id[lu as usize];
        }
        let cs = children[u as usize].clone();
        for &(ch, v) in cs.iter() {
            let mut f = link[u as usize];
            loop {
                if let Some(&w) = edges.get(&key(f, ch)) {
                    link[v as usize] = w;
                    break;
                }
                if f == 0 {
                    link[v as usize] = 0;
                    break;
                }
                f = link[f as usize];
            }
            queue.push(v);
        }
    }

    let n = next() as usize;
    let mut s: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..n {
        s.push(next() as u32);
    }

    let mut mlen: Vec<u32> = vec![0; n];
    let mut mid: Vec<u32> = vec![0; n];

    {
        let mut cur: u32 = 0;
        for j in 0..n {
            let ch = s[n - 1 - j];
            loop {
                if let Some(&w) = edges.get(&key(cur, ch)) {
                    cur = w;
                    break;
                }
                if cur == 0 {
                    break;
                }
                cur = link[cur as usize];
            }
            let i = n - 1 - j;
            mlen[i] = best_len[cur as usize];
            mid[i] = best_id[cur as usize];
        }
    }

    let mut stack: Vec<(u32, usize)> = Vec::new();
    let mut recs: Vec<(u32, usize, usize, usize)> = Vec::new();

    let mut state: u32 = 0;
    let mut depth_bc: u64 = 0;
    let mut err: Option<String> = None;

    let mut i = 0usize;
    while i < n {
        if stack.is_empty() {
            let c = s[i];
            if c == 92 {
                i += 2;
                continue;
            }
            let l = mlen[i] as usize;
            if l > 0 && i + l <= n {
                let id = mid[i];
                let t = (id - 1) / 2 + 1;
                let is_open = (id % 2) == 1;
                if is_open {
                    stack.push((t, i));
                    i += l;
                    continue;
                } else {
                    err = Some(format!("ERROR CLOSE {} {} 0", i, t));
                    break;
                }
            }
            i += 1;
        } else {
            let c = s[i];
            if state == 0 {
                if c == 92 {
                    i += 2;
                    continue;
                }
                if c == 34 {
                    state = 1;
                    i += 1;
                    continue;
                }
                if c == 39 {
                    state = 2;
                    i += 1;
                    continue;
                }
                if c == 47 && i + 1 < n && s[i + 1] == 47 {
                    state = 3;
                    i += 2;
                    continue;
                }
                if c == 47 && i + 1 < n && s[i + 1] == 42 {
                    state = 4;
                    depth_bc = 1;
                    i += 2;
                    continue;
                }
                let l = mlen[i] as usize;
                if l > 0 && i + l <= n {
                    let id = mid[i];
                    let t = (id - 1) / 2 + 1;
                    let is_open = (id % 2) == 1;
                    if is_open {
                        stack.push((t, i));
                        i += l;
                        continue;
                    } else {
                        let top = stack[stack.len() - 1];
                        if top.0 == t {
                            let d = stack.len();
                            recs.push((t, top.1, i + l, d));
                            stack.pop();
                            i += l;
                            continue;
                        } else {
                            err = Some(format!("ERROR CLOSE {} {} {}", i, t, top.0));
                            break;
                        }
                    }
                }
                i += 1;
            } else if state == 1 || state == 2 {
                let q = if state == 1 { 34 } else { 39 };
                if c == 92 {
                    i += 2;
                    continue;
                }
                if c == q {
                    state = 0;
                    i += 1;
                    continue;
                }
                i += 1;
            } else if state == 3 {
                if c == 10 {
                    state = 0;
                    i += 1;
                    continue;
                }
                i += 1;
            } else {
                if c == 47 && i + 1 < n && s[i + 1] == 42 {
                    depth_bc += 1;
                    i += 2;
                    continue;
                }
                if c == 42 && i + 1 < n && s[i + 1] == 47 {
                    depth_bc -= 1;
                    i += 2;
                    if depth_bc == 0 {
                        state = 0;
                    }
                    continue;
                }
                i += 1;
            }
        }
    }

    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());

    if let Some(e) = err {
        writeln!(w, "{}", e).unwrap();
        return;
    }
    if !stack.is_empty() {
        let top = stack[stack.len() - 1];
        writeln!(w, "ERROR OPEN {} {}", top.1, top.0).unwrap();
        return;
    }

    let mut out = String::with_capacity(recs.len() * 20 + 16);
    out.push_str(&recs.len().to_string());
    out.push('\n');
    for r in recs.iter() {
        out.push_str(&r.0.to_string());
        out.push(' ');
        out.push_str(&r.1.to_string());
        out.push(' ');
        out.push_str(&r.2.to_string());
        out.push(' ');
        out.push_str(&r.3.to_string());
        out.push('\n');
    }
    w.write_all(out.as_bytes()).unwrap();
}