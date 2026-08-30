use std::io::{self, Read, Write};
use std::collections::HashMap;

struct Tp {
    k: Vec<u32>,
    v: Vec<u32>,
    p: Vec<u32>,
    l: Vec<u32>,
    r: Vec<u32>,
    s: u64,
}

impl Tp {
    fn new() -> Tp {
        Tp { k: vec![0], v: vec![0], p: vec![0], l: vec![0], r: vec![0], s: 0x9E3779B97F4A7C15 }
    }
    fn rnd(&mut self) -> u32 {
        let mut x = self.s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.s = x;
        (x >> 33) as u32
    }
    fn nn(&mut self, k: u32, v: u32) -> u32 {
        let pr = self.rnd();
        self.k.push(k);
        self.v.push(v);
        self.p.push(pr);
        self.l.push(0);
        self.r.push(0);
        (self.k.len() - 1) as u32
    }
    fn copy(&mut self, t: u32) -> u32 {
        let ti = t as usize;
        let (k, v, p, l, r) = (self.k[ti], self.v[ti], self.p[ti], self.l[ti], self.r[ti]);
        self.k.push(k);
        self.v.push(v);
        self.p.push(p);
        self.l.push(l);
        self.r.push(r);
        (self.k.len() - 1) as u32
    }
    fn get(&self, mut t: u32, k: u32) -> Option<u32> {
        while t != 0 {
            let ti = t as usize;
            if k == self.k[ti] {
                return Some(self.v[ti]);
            } else if k < self.k[ti] {
                t = self.l[ti];
            } else {
                t = self.r[ti];
            }
        }
        None
    }
    fn insert(&mut self, t: u32, k: u32, v: u32) -> u32 {
        if t == 0 {
            return self.nn(k, v);
        }
        let ti = t as usize;
        let tk = self.k[ti];
        if k == tk {
            let n = self.copy(t);
            self.v[n as usize] = v;
            return n;
        }
        if k < tk {
            let sub = self.l[ti];
            let nl = self.insert(sub, k, v);
            if self.p[nl as usize] > self.p[ti] {
                let nt = self.copy(t);
                let nlr = self.r[nl as usize];
                self.l[nt as usize] = nlr;
                self.r[nl as usize] = nt;
                nl
            } else {
                let nt = self.copy(t);
                self.l[nt as usize] = nl;
                nt
            }
        } else {
            let sub = self.r[ti];
            let nr = self.insert(sub, k, v);
            if self.p[nr as usize] > self.p[ti] {
                let nt = self.copy(t);
                let nrl = self.l[nr as usize];
                self.r[nt as usize] = nrl;
                self.l[nr as usize] = nt;
                nr
            } else {
                let nt = self.copy(t);
                self.r[nt as usize] = nr;
                nt
            }
        }
    }
    fn merge(&mut self, a: u32, b: u32) -> u32 {
        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }
        if self.p[a as usize] > self.p[b as usize] {
            let ar = self.r[a as usize];
            let m = self.merge(ar, b);
            let na = self.copy(a);
            self.r[na as usize] = m;
            na
        } else {
            let bl = self.l[b as usize];
            let m = self.merge(a, bl);
            let nb = self.copy(b);
            self.l[nb as usize] = m;
            nb
        }
    }
    fn erase(&mut self, t: u32, k: u32) -> u32 {
        if t == 0 {
            return 0;
        }
        let ti = t as usize;
        let tk = self.k[ti];
        if k == tk {
            let (a, b) = (self.l[ti], self.r[ti]);
            return self.merge(a, b);
        }
        if k < tk {
            let sub = self.l[ti];
            let nl = self.erase(sub, k);
            let nt = self.copy(t);
            self.l[nt as usize] = nl;
            nt
        } else {
            let sub = self.r[ti];
            let nr = self.erase(sub, k);
            let nt = self.copy(t);
            self.r[nt as usize] = nr;
            nt
        }
    }
    fn collect(&self, root: u32, out: &mut Vec<(u32, u32)>) {
        let mut stack: Vec<u32> = Vec::new();
        let mut cur = root;
        while cur != 0 || !stack.is_empty() {
            while cur != 0 {
                stack.push(cur);
                cur = self.l[cur as usize];
            }
            let n = stack.pop().unwrap();
            out.push((self.k[n as usize], self.v[n as usize]));
            cur = self.r[n as usize];
        }
    }
}

struct Op {
    p: u32,
    kind: u8,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
}

fn main() {
    let mut data = Vec::new();
    io::stdin().read_to_end(&mut data).unwrap();
    let mut toks: Vec<(usize, usize)> = Vec::new();
    {
        let n = data.len();
        let mut i = 0usize;
        while i < n {
            while i < n && data[i] <= 32 {
                i += 1;
            }
            if i >= n {
                break;
            }
            let st = i;
            while i < n && data[i] > 32 {
                i += 1;
            }
            toks.push((st, i));
        }
    }

    let mut names: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut nname: u32 = 0;
    let mut syms: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut sym_list: Vec<Vec<u8>> = Vec::new();
    let mut wires: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut wire_list: Vec<Vec<u8>> = Vec::new();
    let mut scopes: HashMap<(u32, u32), u32> = HashMap::new();
    let mut nscope: u32 = 0;

    let mut ti: usize = 0;
    let parse_u = |r: (usize, usize), d: &Vec<u8>| -> u64 {
        let mut x: u64 = 0;
        for j in r.0..r.1 {
            x = x * 10 + (d[j] - b'0') as u64;
        }
        x
    };

    let vcount = parse_u(toks[ti], &data) as usize;
    ti += 1;
    let qcount = parse_u(toks[ti], &data) as usize;
    ti += 1;

    let mut ops: Vec<Op> = Vec::with_capacity(vcount);
    let mut queries: Vec<(u32, u32, u32)> = Vec::with_capacity(qcount);

    macro_rules! sl {
        ($i:expr) => {
            &data[toks[$i].0..toks[$i].1]
        };
    }

    for _ in 0..vcount {
        let p = parse_u(toks[ti], &data) as u32;
        ti += 1;
        let kw = sl!(ti).to_vec();
        ti += 1;
        let mut getname = |idx: usize,
                           names: &mut HashMap<Vec<u8>, u32>,
                           nname: &mut u32,
                           data: &Vec<u8>,
                           toks: &Vec<(usize, usize)>|
         -> u32 {
            let s = &data[toks[idx].0..toks[idx].1];
            if let Some(v) = names.get(s) {
                *v
            } else {
                let id = *nname;
                *nname += 1;
                names.insert(s.to_vec(), id);
                id
            }
        };
        if kw == b"SET" {
            let p1 = getname(ti, &mut names, &mut nname, &data, &toks);
            let r1 = getname(ti + 1, &mut names, &mut nname, &data, &toks);
            let symr = sl!(ti + 2);
            let sid = if let Some(v) = syms.get(symr) {
                *v
            } else {
                let id = sym_list.len() as u32;
                syms.insert(symr.to_vec(), id);
                sym_list.push(symr.to_vec());
                id
            };
            let wr = sl!(ti + 3);
            let wid = if let Some(v) = wires.get(wr) {
                *v
            } else {
                let id = wire_list.len() as u32;
                wires.insert(wr.to_vec(), id);
                wire_list.push(wr.to_vec());
                id
            };
            ti += 4;
            let sc = *scopes.entry((p1, r1)).or_insert_with(|| {
                let id = nscope;
                nscope += 1;
                id
            });
            ops.push(Op { p, kind: 0, a: sc, b: sid, c: wid, d: 0, e: 0 });
        } else if kw == b"DELETE" {
            let p1 = getname(ti, &mut names, &mut nname, &data, &toks);
            let r1 = getname(ti + 1, &mut names, &mut nname, &data, &toks);
            let symr = sl!(ti + 2);
            let sid = if let Some(v) = syms.get(symr) {
                *v
            } else {
                let id = sym_list.len() as u32;
                syms.insert(symr.to_vec(), id);
                sym_list.push(symr.to_vec());
                id
            };
            ti += 3;
            let sc = *scopes.entry((p1, r1)).or_insert_with(|| {
                let id = nscope;
                nscope += 1;
                id
            });
            ops.push(Op { p, kind: 1, a: sc, b: sid, c: 0, d: 0, e: 0 });
        } else if kw == b"MOVE" {
            let sp = getname(ti, &mut names, &mut nname, &data, &toks);
            let sr = getname(ti + 1, &mut names, &mut nname, &data, &toks);
            let ss = sl!(ti + 2);
            let ssid = if let Some(v) = syms.get(ss) {
                *v
            } else {
                let id = sym_list.len() as u32;
                syms.insert(ss.to_vec(), id);
                sym_list.push(ss.to_vec());
                id
            };
            let dp = getname(ti + 3, &mut names, &mut nname, &data, &toks);
            let dr = getname(ti + 4, &mut names, &mut nname, &data, &toks);
            let ds = sl!(ti + 5);
            let dsid = if let Some(v) = syms.get(ds) {
                *v
            } else {
                let id = sym_list.len() as u32;
                syms.insert(ds.to_vec(), id);
                sym_list.push(ds.to_vec());
                id
            };
            ti += 6;
            let sc1 = *scopes.entry((sp, sr)).or_insert_with(|| {
                let id = nscope;
                nscope += 1;
                id
            });
            let sc2 = *scopes.entry((dp, dr)).or_insert_with(|| {
                let id = nscope;
                nscope += 1;
                id
            });
            ops.push(Op { p, kind: 2, a: sc1, b: ssid, c: sc2, d: dsid, e: 0 });
        } else {
            let sp = getname(ti, &mut names, &mut nname, &data, &toks);
            let sr = getname(ti + 1, &mut names, &mut nname, &data, &toks);
            let dp = getname(ti + 2, &mut names, &mut nname, &data, &toks);
            let dr = getname(ti + 3, &mut names, &mut nname, &data, &toks);
            ti += 4;
            let sc1 = *scopes.entry((sp, sr)).or_insert_with(|| {
                let id = nscope;
                nscope += 1;
                id
            });
            let sc2 = *scopes.entry((dp, dr)).or_insert_with(|| {
                let id = nscope;
                nscope += 1;
                id
            });
            ops.push(Op { p, kind: 3, a: sc1, b: 0, c: sc2, d: 0, e: 0 });
        }
    }

    for _ in 0..qcount {
        let o = parse_u(toks[ti], &data) as u32;
        let nv = parse_u(toks[ti + 1], &data) as u32;
        let pn = {
            let s = &data[toks[ti + 2].0..toks[ti + 2].1];
            if let Some(v) = names.get(s) {
                *v
            } else {
                let id = nname;
                nname += 1;
                names.insert(s.to_vec(), id);
                id
            }
        };
        let rn = {
            let s = &data[toks[ti + 3].0..toks[ti + 3].1];
            if let Some(v) = names.get(s) {
                *v
            } else {
                let id = nname;
                nname += 1;
                names.insert(s.to_vec(), id);
                id
            }
        };
        ti += 4;
        let sc = *scopes.entry((pn, rn)).or_insert_with(|| {
            let id = nscope;
            nscope += 1;
            id
        });
        queries.push((o, nv, sc));
    }

    let ns = sym_list.len();
    let mut order: Vec<u32> = (0..ns as u32).collect();
    order.sort_by(|a, b| sym_list[*a as usize].cmp(&sym_list[*b as usize]));
    let mut rank: Vec<u32> = vec![0; ns];
    let mut by_rank: Vec<u32> = vec![0; ns];
    for (i, id) in order.iter().enumerate() {
        rank[*id as usize] = i as u32;
        by_rank[i] = *id;
    }

    let mut tp = Tp::new();
    let mut roots: Vec<u32> = vec![0; vcount + 1];

    for i in 0..vcount {
        let op = &ops[i];
        let outer = roots[op.p as usize];
        let res;
        match op.kind {
            0 => {
                let sc = op.a;
                let inner = tp.get(outer, sc).unwrap_or(0);
                let ni = tp.insert(inner, rank[op.b as usize], op.c);
                res = tp.insert(outer, sc, ni);
            }
            1 => {
                let sc = op.a;
                let inner = tp.get(outer, sc).unwrap_or(0);
                let rk = rank[op.b as usize];
                if inner == 0 || tp.get(inner, rk).is_none() {
                    res = outer;
                } else {
                    let ni = tp.erase(inner, rk);
                    res = if ni == 0 { tp.erase(outer, sc) } else { tp.insert(outer, sc, ni) };
                }
            }
            2 => {
                let ssc = op.a;
                let dsc = op.c;
                let srk = rank[op.b as usize];
                let drk = rank[op.d as usize];
                let inner_s = tp.get(outer, ssc).unwrap_or(0);
                let w = tp.get(inner_s, srk);
                match w {
                    None => {
                        res = outer;
                    }
                    Some(wv) => {
                        if ssc == dsc && srk == drk {
                            res = outer;
                        } else if ssc == dsc {
                            let mut inner = tp.erase(inner_s, srk);
                            inner = tp.insert(inner, drk, wv);
                            res = tp.insert(outer, ssc, inner);
                        } else {
                            let is2 = tp.erase(inner_s, srk);
                            let outer2 = if is2 == 0 {
                                tp.erase(outer, ssc)
                            } else {
                                tp.insert(outer, ssc, is2)
                            };
                            let inner_d = tp.get(outer2, dsc).unwrap_or(0);
                            let id2 = tp.insert(inner_d, drk, wv);
                            res = tp.insert(outer2, dsc, id2);
                        }
                    }
                }
            }
            _ => {
                let ssc = op.a;
                let dsc = op.c;
                if ssc == dsc {
                    res = outer;
                } else {
                    let sroot = tp.get(outer, ssc).unwrap_or(0);
                    if sroot == 0 {
                        res = if tp.get(outer, dsc).is_some() {
                            tp.erase(outer, dsc)
                        } else {
                            outer
                        };
                    } else {
                        res = tp.insert(outer, dsc, sroot);
                    }
                }
            }
        }
        roots[i + 1] = res;
    }

    let stdout = io::stdout();
    let mut out = io::BufWriter::with_capacity(1 << 20, stdout.lock());
    let mut va: Vec<(u32, u32)> = Vec::new();
    let mut vb: Vec<(u32, u32)> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();

    for q in &queries {
        let (ov, nv, sc) = *q;
        let ro = roots[ov as usize];
        let rn = roots[nv as usize];
        let ia = tp.get(ro, sc).unwrap_or(0);
        let ib = tp.get(rn, sc).unwrap_or(0);
        va.clear();
        vb.clear();
        tp.collect(ia, &mut va);
        tp.collect(ib, &mut vb);
        buf.clear();
        let mut cnt: u64 = 0;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < va.len() || j < vb.len() {
            if j >= vb.len() || (i < va.len() && va[i].0 < vb[j].0) {
                let (rk, w) = va[i];
                buf.extend_from_slice(b"REMOVED ");
                buf.extend_from_slice(&sym_list[by_rank[rk as usize] as usize]);
                buf.push(b' ');
                buf.extend_from_slice(&wire_list[w as usize]);
                buf.push(b'\n');
                i += 1;
            } else if i >= va.len() || vb[j].0 < va[i].0 {
                let (rk, w) = vb[j];
                buf.extend_from_slice(b"ADDED ");
                buf.extend_from_slice(&sym_list[by_rank[rk as usize] as usize]);
                buf.push(b' ');
                buf.extend_from_slice(&wire_list[w as usize]);
                buf.push(b'\n');
                j += 1;
            } else {
                let (rk, w1) = va[i];
                let w2 = vb[j].1;
                if w1 == w2 {
                    buf.extend_from_slice(b"UNCHANGED ");
                    buf.extend_from_slice(&sym_list[by_rank[rk as usize] as usize]);
                    buf.push(b' ');
                    buf.extend_from_slice(&wire_list[w1 as usize]);
                    buf.push(b'\n');
                } else {
                    buf.extend_from_slice(b"CHANGED ");
                    buf.extend_from_slice(&sym_list[by_rank[rk as usize] as usize]);
                    buf.push(b' ');
                    buf.extend_from_slice(&wire_list[w1 as usize]);
                    buf.push(b' ');
                    buf.extend_from_slice(&wire_list[w2 as usize]);
                    buf.push(b'\n');
                }
                i += 1;
                j += 1;
            }
            cnt += 1;
        }
        let mut nb = Vec::new();
        let mut c = cnt;
        if c == 0 {
            nb.push(b'0');
        } else {
            let mut tmp = Vec::new();
            while c > 0 {
                tmp.push(b'0' + (c % 10) as u8);
                c /= 10;
            }
            tmp.reverse();
            nb = tmp;
        }
        out.write_all(&nb).unwrap();
        out.write_all(b"\n").unwrap();
        out.write_all(&buf).unwrap();
    }
    out.flush().unwrap();
}