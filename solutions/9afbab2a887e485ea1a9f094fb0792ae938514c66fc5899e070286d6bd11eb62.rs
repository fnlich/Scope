use std::collections::HashMap;
use std::io::{self, Read, Write};

const PRIM: u8 = 0;
const REFN: u8 = 1;
const ARR: u8 = 2;
const MP: u8 = 3;
const ST: u8 = 4;
const UN: u8 = 5;

fn isname(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() {
        return false;
    }
    if !(b[0] >= b'a' && b[0] <= b'z') {
        return false;
    }
    for i in 1..b.len() {
        let c = b[i];
        if !((c >= b'a' && c <= b'z') || (c >= b'0' && c <= b'9') || c == b'_') {
            return false;
        }
    }
    true
}

struct Fr {
    kind: u8,
    kids: Vec<u32>,
    fnames: Vec<u32>,
}

struct P<'a> {
    t: Vec<&'a str>,
    pos: usize,
    kind: Vec<u8>,
    a: Vec<u32>,
    b: Vec<u32>,
    kids: Vec<u32>,
    map: HashMap<&'a str, u32>,
    nn: u32,
}

impl<'a> P<'a> {
    fn nx(&mut self) -> Option<&'a str> {
        if self.pos < self.t.len() {
            let s = self.t[self.pos];
            self.pos += 1;
            Some(s)
        } else {
            None
        }
    }
    fn exp(&mut self, s: &str) -> bool {
        match self.nx() {
            Some(x) => x == s,
            None => false,
        }
    }
    fn intern(&mut self, s: &'a str) -> u32 {
        if let Some(&v) = self.map.get(s) {
            v
        } else {
            let v = self.nn;
            self.nn += 1;
            self.map.insert(s, v);
            v
        }
    }
    fn nd(&mut self, k: u8, x: u32, y: u32) -> u32 {
        self.kind.push(k);
        self.a.push(x);
        self.b.push(y);
        (self.kind.len() - 1) as u32
    }
    fn pushkids(&mut self, v: &[u32]) -> (u32, u32) {
        let s = self.kids.len() as u32;
        for x in v {
            self.kids.push(*x);
        }
        (s, v.len() as u32)
    }
    fn parse_type(&mut self) -> Option<u32> {
        let mut stack: Vec<Fr> = Vec::new();
        'outer: loop {
            let tok = match self.nx() {
                Some(x) => x,
                None => return None,
            };
            let mut val: u32;
            match tok {
                "INT" | "STRING" | "BOOL" => {
                    val = self.nd(PRIM, 0, 0);
                }
                "ARRAY" => {
                    if !self.exp("<") {
                        return None;
                    }
                    stack.push(Fr {
                        kind: ARR,
                        kids: Vec::new(),
                        fnames: Vec::new(),
                    });
                    continue 'outer;
                }
                "MAP" => {
                    if !self.exp("<") {
                        return None;
                    }
                    stack.push(Fr {
                        kind: MP,
                        kids: Vec::new(),
                        fnames: Vec::new(),
                    });
                    continue 'outer;
                }
                "UNION" => {
                    if !self.exp("<") {
                        return None;
                    }
                    stack.push(Fr {
                        kind: UN,
                        kids: Vec::new(),
                        fnames: Vec::new(),
                    });
                    continue 'outer;
                }
                "STRUCT" => {
                    if !self.exp("<") {
                        return None;
                    }
                    let nm = match self.nx() {
                        Some(x) => x,
                        None => return None,
                    };
                    if !isname(nm) {
                        return None;
                    }
                    let id = self.intern(nm);
                    if !self.exp(":") {
                        return None;
                    }
                    let mut f = Fr {
                        kind: ST,
                        kids: Vec::new(),
                        fnames: Vec::new(),
                    };
                    f.fnames.push(id);
                    stack.push(f);
                    continue 'outer;
                }
                "TYPE" | "ROOT" | "<" | ">" | "," | ":" | "=" | ";" => {
                    return None;
                }
                _ => {
                    if !isname(tok) {
                        return None;
                    }
                    let id = self.intern(tok);
                    val = self.nd(REFN, id, 0);
                }
            }
            loop {
                if stack.is_empty() {
                    return Some(val);
                }
                let ti = stack.len() - 1;
                let k = stack[ti].kind;
                if k == ARR {
                    if !self.exp(">") {
                        return None;
                    }
                    stack.pop();
                    val = self.nd(ARR, val, 0);
                } else if k == MP {
                    if stack[ti].kids.is_empty() {
                        stack[ti].kids.push(val);
                        if !self.exp(",") {
                            return None;
                        }
                        continue 'outer;
                    } else {
                        if !self.exp(">") {
                            return None;
                        }
                        let kk = stack[ti].kids[0];
                        stack.pop();
                        val = self.nd(MP, kk, val);
                    }
                } else if k == ST {
                    stack[ti].kids.push(val);
                    let nxt = match self.nx() {
                        Some(x) => x,
                        None => return None,
                    };
                    if nxt == "," {
                        let nm = match self.nx() {
                            Some(x) => x,
                            None => return None,
                        };
                        if !isname(nm) {
                            return None;
                        }
                        let id = self.intern(nm);
                        stack[ti].fnames.push(id);
                        if !self.exp(":") {
                            return None;
                        }
                        continue 'outer;
                    } else if nxt == ">" {
                        let f = stack.pop().unwrap();
                        if f.kids.is_empty() {
                            return None;
                        }
                        let mut sn = f.fnames.clone();
                        sn.sort();
                        for w in sn.windows(2) {
                            if w[0] == w[1] {
                                return None;
                            }
                        }
                        let (s0, l0) = self.pushkids(&f.kids);
                        val = self.nd(ST, s0, l0);
                    } else {
                        return None;
                    }
                } else {
                    stack[ti].kids.push(val);
                    let nxt = match self.nx() {
                        Some(x) => x,
                        None => return None,
                    };
                    if nxt == "," {
                        continue 'outer;
                    } else if nxt == ">" {
                        let f = stack.pop().unwrap();
                        if f.kids.len() < 2 {
                            return None;
                        }
                        let (s0, l0) = self.pushkids(&f.kids);
                        val = self.nd(UN, s0, l0);
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

fn run<'a>(toks: &[&'a str]) -> Option<String> {
    if toks.is_empty() {
        return None;
    }
    if toks[0].parse::<i64>().is_err() {
        return None;
    }
    let mut p = P {
        t: toks[1..].to_vec(),
        pos: 0,
        kind: Vec::new(),
        a: Vec::new(),
        b: Vec::new(),
        kids: Vec::new(),
        map: HashMap::new(),
        nn: 0,
    };
    let mut defs: Vec<(u32, u32, usize, usize)> = Vec::new();
    let mut roots: Vec<(u32, usize, usize)> = Vec::new();
    let mut seen = false;
    while p.pos < p.t.len() {
        let tk = match p.nx() {
            Some(x) => x,
            None => return None,
        };
        if tk == "TYPE" {
            if seen {
                return None;
            }
            let nm = match p.nx() {
                Some(x) => x,
                None => return None,
            };
            if !isname(nm) {
                return None;
            }
            let id = p.intern(nm);
            if !p.exp("=") {
                return None;
            }
            let s0 = p.kind.len();
            let nd = match p.parse_type() {
                Some(x) => x,
                None => return None,
            };
            let e0 = p.kind.len();
            if !p.exp(";") {
                return None;
            }
            defs.push((id, nd, s0, e0));
        } else if tk == "ROOT" {
            seen = true;
            if !p.exp("=") {
                return None;
            }
            let s0 = p.kind.len();
            let nd = match p.parse_type() {
                Some(x) => x,
                None => return None,
            };
            let e0 = p.kind.len();
            if !p.exp(";") {
                return None;
            }
            roots.push((nd, s0, e0));
        } else {
            return None;
        }
    }
    if roots.is_empty() {
        return None;
    }

    let d = defs.len();
    let mut nm2d: HashMap<u32, usize> = HashMap::new();
    for i in 0..d {
        if nm2d.insert(defs[i].0, i).is_some() {
            return None;
        }
    }
    let nn = p.kind.len();
    for i in 0..nn {
        if p.kind[i] == REFN {
            let id = p.a[i];
            match nm2d.get(&id) {
                Some(&x) => {
                    p.b[i] = x as u32;
                }
                None => return None,
            }
        }
    }
    let mut defroot = vec![0u32; d];
    for i in 0..d {
        defroot[i] = defs[i].1;
    }

    let mut memo = vec![0u8; d];
    for i in 0..nn {
        if p.kind[i] == MP {
            let start = p.a[i] as usize;
            let mut path: Vec<usize> = Vec::new();
            let mut cur = start;
            let res;
            loop {
                let k = p.kind[cur];
                if k == PRIM {
                    res = true;
                    break;
                } else if k == REFN {
                    let dd = p.b[cur] as usize;
                    if memo[dd] == 1 {
                        res = true;
                        break;
                    } else if memo[dd] == 2 || memo[dd] == 3 {
                        res = false;
                        break;
                    } else {
                        memo[dd] = 3;
                        path.push(dd);
                        cur = defroot[dd] as usize;
                    }
                } else {
                    res = false;
                    break;
                }
            }
            for x in path.iter() {
                memo[*x] = if res { 1 } else { 2 };
            }
            if !res {
                return None;
            }
        }
    }

    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); d];
    let mut adju: Vec<Vec<u32>> = vec![Vec::new(); d];
    for di in 0..d {
        let mut stk: Vec<(usize, bool)> = vec![(defroot[di] as usize, false)];
        while let Some((nd, g)) = stk.pop() {
            let k = p.kind[nd];
            if k == PRIM {
            } else if k == REFN {
                let e = p.b[nd];
                adj[di].push(e);
                if !g {
                    adju[di].push(e);
                }
            } else if k == ARR {
                stk.push((p.a[nd] as usize, true));
            } else if k == MP {
                stk.push((p.a[nd] as usize, g));
                stk.push((p.b[nd] as usize, true));
            } else {
                let s0 = p.a[nd] as usize;
                let ln = p.b[nd] as usize;
                for j in s0..s0 + ln {
                    stk.push((p.kids[j] as usize, g));
                }
            }
        }
    }

    let mut col = vec![0u8; d];
    let mut st: Vec<(usize, usize)> = Vec::new();
    for s in 0..d {
        if col[s] != 0 {
            continue;
        }
        col[s] = 1;
        st.push((s, 0));
        while !st.is_empty() {
            let (u, i) = *st.last().unwrap();
            if i < adju[u].len() {
                st.last_mut().unwrap().1 += 1;
                let v = adju[u][i] as usize;
                if col[v] == 1 {
                    return None;
                } else if col[v] == 0 {
                    col[v] = 1;
                    st.push((v, 0));
                }
            } else {
                st.pop();
                col[u] = 2;
            }
        }
    }

    let mut inf = vec![false; d];
    let mut c2 = vec![0u8; d];
    let mut st2: Vec<(usize, usize)> = Vec::new();
    for s in 0..d {
        if c2[s] != 0 {
            continue;
        }
        c2[s] = 1;
        st2.push((s, 0));
        while !st2.is_empty() {
            let (u, i) = *st2.last().unwrap();
            if i < adj[u].len() {
                st2.last_mut().unwrap().1 += 1;
                let v = adj[u][i] as usize;
                if c2[v] == 1 {
                    inf[u] = true;
                } else if c2[v] == 2 {
                    if inf[v] {
                        inf[u] = true;
                    }
                } else {
                    c2[v] = 1;
                    st2.push((v, 0));
                }
            } else {
                st2.pop();
                c2[u] = 2;
                if inf[u] {
                    if let Some(t) = st2.last() {
                        let pu = t.0;
                        inf[pu] = true;
                    }
                }
            }
        }
    }

    let mut ni = vec![false; nn];
    for i in 0..nn {
        let k = p.kind[i];
        ni[i] = if k == PRIM {
            false
        } else if k == REFN {
            inf[p.b[i] as usize]
        } else if k == ARR {
            ni[p.a[i] as usize]
        } else if k == MP {
            ni[p.a[i] as usize] || ni[p.b[i] as usize]
        } else {
            let s0 = p.a[i] as usize;
            let ln = p.b[i] as usize;
            let mut r = false;
            for j in s0..s0 + ln {
                if ni[p.kids[j] as usize] {
                    r = true;
                }
            }
            r
        };
    }

    let mut depth = vec![0i64; nn];
    let mut dd = vec![0i64; d];
    let mut c3 = vec![0u8; d];
    let mut st3: Vec<(usize, usize)> = Vec::new();
    for s in 0..d {
        if inf[s] || c3[s] != 0 {
            continue;
        }
        c3[s] = 1;
        st3.push((s, 0));
        while !st3.is_empty() {
            let (u, i) = *st3.last().unwrap();
            if i < adj[u].len() {
                st3.last_mut().unwrap().1 += 1;
                let v = adj[u][i] as usize;
                if c3[v] == 0 {
                    c3[v] = 1;
                    st3.push((v, 0));
                }
            } else {
                st3.pop();
                c3[u] = 2;
                let s0 = defs[u].2;
                let e0 = defs[u].3;
                for x in s0..e0 {
                    let k = p.kind[x];
                    depth[x] = if k == PRIM {
                        0
                    } else if k == REFN {
                        dd[p.b[x] as usize]
                    } else if k == ARR {
                        1 + depth[p.a[x] as usize]
                    } else if k == MP {
                        let m1 = depth[p.a[x] as usize];
                        let m2 = depth[p.b[x] as usize];
                        1 + if m1 > m2 { m1 } else { m2 }
                    } else {
                        let ks = p.a[x] as usize;
                        let ln = p.b[x] as usize;
                        let mut m = 0i64;
                        for j in ks..ks + ln {
                            let z = depth[p.kids[j] as usize];
                            if z > m {
                                m = z;
                            }
                        }
                        1 + m
                    };
                }
                dd[u] = depth[defs[u].1 as usize];
            }
        }
    }

    let mut out = String::new();
    for r in 0..roots.len() {
        let (rt, s0, e0) = roots[r];
        if ni[rt as usize] {
            out.push_str("VALID INFINITE\n");
        } else {
            for x in s0..e0 {
                let k = p.kind[x];
                depth[x] = if k == PRIM {
                    0
                } else if k == REFN {
                    dd[p.b[x] as usize]
                } else if k == ARR {
                    1 + depth[p.a[x] as usize]
                } else if k == MP {
                    let m1 = depth[p.a[x] as usize];
                    let m2 = depth[p.b[x] as usize];
                    1 + if m1 > m2 { m1 } else { m2 }
                } else {
                    let ks = p.a[x] as usize;
                    let ln = p.b[x] as usize;
                    let mut m = 0i64;
                    for j in ks..ks + ln {
                        let z = depth[p.kids[j] as usize];
                        if z > m {
                            m = z;
                        }
                    }
                    1 + m
                };
            }
            out.push_str("VALID FINITE ");
            out.push_str(&depth[rt as usize].to_string());
            out.push('\n');
        }
    }
    Some(out)
}

fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let toks: Vec<&str> = s.split_ascii_whitespace().collect();
    let stdout = io::stdout();
    let mut o = io::BufWriter::new(stdout.lock());
    match run(&toks) {
        Some(x) => {
            o.write_all(x.as_bytes()).unwrap();
        }
        None => {
            o.write_all(b"INVALID\n").unwrap();
        }
    }
}