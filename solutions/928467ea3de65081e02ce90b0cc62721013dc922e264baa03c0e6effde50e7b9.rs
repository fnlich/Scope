use std::io::{self, Read, Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;

struct Seg {
    n: usize,
    mn: Vec<i64>,
    sm: Vec<i64>,
    lz: Vec<i64>,
}

impl Seg {
    fn new(n: usize) -> Seg {
        Seg { n, mn: vec![0; 4 * n + 4], sm: vec![0; 4 * n + 4], lz: vec![0; 4 * n + 4] }
    }
    fn pull(&mut self, node: usize) {
        let a = node * 2;
        let b = node * 2 + 1;
        if self.mn[a] < self.mn[b] {
            self.mn[node] = self.mn[a];
            self.sm[node] = self.sm[a];
        } else if self.mn[b] < self.mn[a] {
            self.mn[node] = self.mn[b];
            self.sm[node] = self.sm[b];
        } else {
            self.mn[node] = self.mn[a];
            self.sm[node] = self.sm[a] + self.sm[b];
        }
    }
    fn push(&mut self, node: usize) {
        let v = self.lz[node];
        if v != 0 {
            for c in [node * 2, node * 2 + 1] {
                self.mn[c] += v;
                self.lz[c] += v;
            }
            self.lz[node] = 0;
        }
    }
    fn upd(&mut self, node: usize, l: usize, r: usize, ql: usize, qr: usize, v: i64) {
        if qr < l || r < ql {
            return;
        }
        if ql <= l && r <= qr {
            self.mn[node] += v;
            self.lz[node] += v;
            return;
        }
        self.push(node);
        let m = (l + r) / 2;
        self.upd(node * 2, l, m, ql, qr, v);
        self.upd(node * 2 + 1, m + 1, r, ql, qr, v);
        self.pull(node);
    }
    fn set_point(&mut self, node: usize, l: usize, r: usize, pos: usize, v: i64) {
        if l == r {
            self.sm[node] = v;
            return;
        }
        self.push(node);
        let m = (l + r) / 2;
        if pos <= m {
            self.set_point(node * 2, l, m, pos, v);
        } else {
            self.set_point(node * 2 + 1, m + 1, r, pos, v);
        }
        self.pull(node);
    }
    fn query(&mut self, node: usize, l: usize, r: usize, ql: usize, qr: usize) -> (i64, i64) {
        if qr < l || r < ql {
            return (i64::MAX, 0);
        }
        if ql <= l && r <= qr {
            return (self.mn[node], self.sm[node]);
        }
        self.push(node);
        let m = (l + r) / 2;
        let a = self.query(node * 2, l, m, ql, qr);
        let b = self.query(node * 2 + 1, m + 1, r, ql, qr);
        if a.0 < b.0 {
            a
        } else if b.0 < a.0 {
            b
        } else {
            (a.0, a.1 + b.1)
        }
    }
    fn range_add(&mut self, ql: usize, qr: usize, v: i64) {
        let n = self.n;
        self.upd(1, 0, n - 1, ql, qr, v);
    }
    fn point(&mut self, pos: usize, v: i64) {
        let n = self.n;
        self.set_point(1, 0, n - 1, pos, v);
    }
    fn ask(&mut self, ql: usize, qr: usize) -> i64 {
        let n = self.n;
        let (mn, sm) = self.query(1, 0, n - 1, ql, qr);
        if mn == 0 {
            sm
        } else {
            0
        }
    }
}

struct Scanner {
    buf: Vec<u8>,
    pos: usize,
}

impl Scanner {
    fn new() -> Scanner {
        let mut s = Vec::new();
        io::stdin().read_to_end(&mut s).unwrap();
        Scanner { buf: s, pos: 0 }
    }
    fn skip(&mut self) {
        while self.pos < self.buf.len() && (self.buf[self.pos] as char).is_ascii_whitespace() {
            self.pos += 1;
        }
    }
    fn word(&mut self) -> u8 {
        self.skip();
        let c = self.buf[self.pos];
        while self.pos < self.buf.len() && !(self.buf[self.pos] as char).is_ascii_whitespace() {
            self.pos += 1;
        }
        c
    }
    fn num(&mut self) -> i64 {
        self.skip();
        let mut neg = false;
        if self.buf[self.pos] == b'-' {
            neg = true;
            self.pos += 1;
        }
        let mut v: i64 = 0;
        while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_digit() {
            v = v * 10 + (self.buf[self.pos] - b'0') as i64;
            self.pos += 1;
        }
        if neg {
            -v
        } else {
            v
        }
    }
}

fn main() {
    let mut sc = Scanner::new();
    let n = sc.num() as usize;
    let c = sc.num() as usize;
    let a = sc.num() as usize;
    let q = sc.num() as usize;

    let mut par = vec![0usize; c + 1];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); c + 1];
    let mut roots: Vec<usize> = Vec::new();
    for i in 1..=c {
        let p = sc.num() as usize;
        par[i] = p;
        if p == 0 {
            roots.push(i);
        } else {
            children[p].push(i);
        }
    }

    let mut tin = vec![0usize; c + 1];
    let mut tout = vec![0usize; c + 1];
    {
        let mut timer = 0usize;
        let mut stack: Vec<(usize, bool)> = Vec::new();
        for &r in roots.iter().rev() {
            stack.push((r, false));
        }
        while let Some((v, st)) = stack.pop() {
            if !st {
                tin[v] = timer;
                timer += 1;
                stack.push((v, true));
                for &ch in children[v].iter().rev() {
                    stack.push((ch, false));
                }
            } else {
                tout[v] = timer - 1;
            }
        }
    }

    let mut rq = vec![0i64; n + 1];
    let mut re = vec![0i64; n + 1];
    let mut ru = vec![0i64; n + 1];
    let mut rg = vec![0usize; n + 1];
    let mut rk = vec![0i64; n + 1];
    for i in 1..=n {
        rq[i] = sc.num();
        re[i] = sc.num();
        ru[i] = sc.num();
        rg[i] = sc.num() as usize;
        rk[i] = sc.num();
    }

    let mut acat = vec![0usize; a + 1];
    let mut astock = vec![0i64; a + 1];
    let mut aexp = vec![0i64; a + 1];
    for i in 1..=a {
        acat[i] = sc.num() as usize;
        astock[i] = sc.num();
        aexp[i] = sc.num();
    }

    let mut t = sc.num();
    let mut live = sc.num() != 0;
    let mut gaddons = sc.num() != 0;
    let mut selected = sc.num() as usize;

    let mut seg = Seg::new(c);
    let mut cnt = vec![0i64; c + 1];
    let mut active = vec![false; a + 1];
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();

    for i in 1..=a {
        if acat[i] >= 1 && acat[i] <= c && astock[i] > 0 && aexp[i] >= t {
            active[i] = true;
            cnt[acat[i]] += 1;
            heap.push(Reverse((aexp[i], i)));
        }
    }
    for cc in 1..=c {
        if cnt[cc] != 0 {
            seg.point(tin[cc], cnt[cc]);
        }
    }

    let mut enabled = vec![true; c + 1];

    let mut out: Vec<(usize, usize, u8)> = Vec::new();

    for op_idx in 1..=q {
        let op = sc.word();
        match op {
            b'T' => {
                let nt = sc.num();
                t = nt;
                loop {
                    let top = match heap.peek() {
                        Some(&Reverse((e, _))) => e,
                        None => break,
                    };
                    if top >= t {
                        break;
                    }
                    let Reverse((e, id)) = heap.pop().unwrap();
                    if active[id] && aexp[id] == e {
                        active[id] = false;
                        let cc = acat[id];
                        cnt[cc] -= 1;
                        seg.point(tin[cc], cnt[cc]);
                    }
                }
            }
            b'L' => {
                live = sc.num() != 0;
            }
            b'G' => {
                gaddons = sc.num() != 0;
            }
            b'C' => {
                let cc = sc.num() as usize;
                let b = sc.num() != 0;
                if cc >= 1 && cc <= c && enabled[cc] != b {
                    enabled[cc] = b;
                    let d = if b { -1i64 } else { 1i64 };
                    seg.range_add(tin[cc], tout[cc], d);
                }
            }
            b'A' => {
                let id = sc.num() as usize;
                let st = sc.num();
                let ex = sc.num();
                let cc = acat[id];
                if active[id] {
                    active[id] = false;
                    cnt[cc] -= 1;
                    seg.point(tin[cc], cnt[cc]);
                }
                astock[id] = st;
                aexp[id] = ex;
                if cc >= 1 && cc <= c && st > 0 && ex >= t {
                    active[id] = true;
                    cnt[cc] += 1;
                    seg.point(tin[cc], cnt[cc]);
                    heap.push(Reverse((ex, id)));
                }
            }
            b'R' => {
                let i = sc.num() as usize;
                rq[i] = sc.num();
                re[i] = sc.num();
                ru[i] = sc.num();
                rk[i] = sc.num();
            }
            b'S' => {
                selected = sc.num() as usize;
            }
            b'K' => {
                let i = sc.num() as usize;
                if live && i >= 1 && i <= n {
                    if selected == i {
                        if gaddons {
                            let g = rg[i];
                            let avail = if g == 0 || g > c {
                                0
                            } else {
                                seg.ask(tin[g], tout[g])
                            };
                            if avail >= rk[i] {
                                out.push((op_idx, i, 1));
                            }
                        }
                    } else {
                        if rq[i] > 0 && re[i] >= t && ru[i] == 1 {
                            out.push((op_idx, i, 0));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());
    let mut s = String::with_capacity(out.len() * 16 + 16);
    s.push_str(&out.len().to_string());
    s.push('\n');
    for &(o, i, m) in out.iter() {
        s.push_str(&o.to_string());
        s.push(' ');
        s.push_str(&i.to_string());
        s.push(' ');
        s.push_str(if m == 0 { "BASE" } else { "ADDON" });
        s.push('\n');
    }
    w.write_all(s.as_bytes()).unwrap();
}