use std::io::{self, Read, Write};

const NONE: usize = usize::MAX;

struct Trace {
    n: usize,
    x: Vec<i128>,
    dsum: Vec<i128>,
    ssum: Vec<i128>,
    term: Vec<usize>,
    tree: Vec<i128>,
    size: usize,
}

impl Trace {
    fn new(d: Vec<i128>, x: Vec<i128>, f: i128) -> Trace {
        let n = d.len();
        let mut dsum = vec![0i128; n + 1];
        let mut ssum = vec![0i128; n + 1];
        for i in 0..n {
            dsum[i + 1] = dsum[i] + d[i];
            let add = if x[i] > 0 { x[i] } else { 0 };
            ssum[i + 1] = ssum[i] + add;
        }
        let mut term = vec![NONE; n + 1];
        term[n] = NONE;
        for i in (0..n).rev() {
            if x[i] == 0 || x[i] == -2 {
                term[i] = i;
            } else {
                term[i] = term[i + 1];
            }
        }
        let mut size = 1usize;
        while size < n.max(1) {
            size *= 2;
        }
        let mut tree = vec![i128::MIN; 2 * size];
        for i in 0..n {
            tree[size + i] = dsum[i + 1] - f * ssum[i];
        }
        for i in (1..size).rev() {
            tree[i] = if tree[2 * i] > tree[2 * i + 1] {
                tree[2 * i]
            } else {
                tree[2 * i + 1]
            };
        }
        Trace {
            n,
            x,
            dsum,
            ssum,
            term,
            tree,
            size,
        }
    }

    fn first_gt(&self, l: usize, c: i128) -> usize {
        if l >= self.n {
            return NONE;
        }
        self.descend(1, 0, self.size, l, c)
    }

    fn descend(&self, node: usize, nl: usize, nr: usize, l: usize, c: i128) -> usize {
        if nr <= l {
            return NONE;
        }
        if self.tree[node] <= c {
            return NONE;
        }
        if nr - nl == 1 {
            if nl < self.n {
                return nl;
            }
            return NONE;
        }
        let mid = (nl + nr) / 2;
        let r = self.descend(2 * node, nl, mid, l, c);
        if r != NONE {
            return r;
        }
        self.descend(2 * node + 1, mid, nr, l, c)
    }
}

fn parse_all(s: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    for &b in s {
        if b == 0x20 || (b >= 0x09 && b <= 0x0D) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(b);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn to_int(t: &[u8]) -> i128 {
    let mut neg = false;
    let mut i = 0;
    if t[0] == b'-' {
        neg = true;
        i = 1;
    } else if t[0] == b'+' {
        i = 1;
    }
    let mut v: i128 = 0;
    while i < t.len() {
        v = v * 10 + (t[i] - b'0') as i128;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

fn main() {
    let mut inp = Vec::new();
    io::stdin().read_to_end(&mut inp).unwrap();
    let toks = parse_all(&inp);
    let mut p = 0usize;
    let nr = to_int(&toks[p]) as usize;
    p += 1;
    let nw = to_int(&toks[p]) as usize;
    p += 1;
    let q = to_int(&toks[p]) as usize;
    p += 1;
    let ar = to_int(&toks[p]) as i64;
    p += 1;
    let aw = to_int(&toks[p]) as i64;
    p += 1;
    let f = to_int(&toks[p]);
    p += 1;

    let mut dr = Vec::with_capacity(nr);
    let mut xr = Vec::with_capacity(nr);
    for _ in 0..nr {
        dr.push(to_int(&toks[p]));
        p += 1;
        xr.push(to_int(&toks[p]));
        p += 1;
    }
    let mut dw = Vec::with_capacity(nw);
    let mut xw = Vec::with_capacity(nw);
    for _ in 0..nw {
        dw.push(to_int(&toks[p]));
        p += 1;
        xw.push(to_int(&toks[p]));
        p += 1;
    }

    let tr = Trace::new(dr, xr, f);
    let tw = Trace::new(dw, xw, f);

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for _ in 0..q {
        let c = toks[p][0];
        p += 1;
        let l = to_int(&toks[p]) as usize;
        p += 1;
        let k = to_int(&toks[p]);
        p += 1;
        let t = to_int(&toks[p]);
        p += 1;

        let (tt, avail) = if c == b'R' { (&tr, ar) } else { (&tw, aw) };
        if avail == 0 {
            out.write_all(b"UNAVAILABLE 0\n").unwrap();
            continue;
        }
        if k == 0 {
            out.write_all(b"OK 0\n").unwrap();
            continue;
        }
        let l0 = l - 1;
        let n = tt.n;
        if l0 >= n {
            out.write_all(b"EXHAUSTED 0\n").unwrap();
            continue;
        }
        let cval = t + tt.dsum[l0] - f * tt.ssum[l0];
        let ti = tt.first_gt(l0, cval);
        let term = tt.term[l0];
        let target = k + tt.ssum[l0];
        let mut kidx = NONE;
        {
            let mut lo = l0 + 1;
            let mut hi = n + 1;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if tt.ssum[mid] >= target {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            if lo <= n {
                kidx = lo - 1;
            }
        }

        if ti != NONE && ti <= kidx && ti <= term {
            let bytes = tt.ssum[ti] - tt.ssum[l0];
            if bytes > 0 {
                out.write_all(format!("PARTIAL {}\n", bytes).as_bytes()).unwrap();
            } else {
                out.write_all(b"TIMEOUT 0\n").unwrap();
            }
        } else if kidx != NONE && kidx < term {
            out.write_all(format!("OK {}\n", k).as_bytes()).unwrap();
        } else if term != NONE {
            let bytes = tt.ssum[term] - tt.ssum[l0];
            if bytes > 0 {
                out.write_all(format!("PARTIAL {}\n", bytes).as_bytes()).unwrap();
            } else if tt.x[term] == 0 {
                out.write_all(b"EOF 0\n").unwrap();
            } else {
                out.write_all(b"ERROR 0\n").unwrap();
            }
        } else {
            let bytes = tt.ssum[n] - tt.ssum[l0];
            if bytes > 0 {
                out.write_all(format!("PARTIAL {}\n", bytes).as_bytes()).unwrap();
            } else {
                out.write_all(b"EXHAUSTED 0\n").unwrap();
            }
        }
    }
}