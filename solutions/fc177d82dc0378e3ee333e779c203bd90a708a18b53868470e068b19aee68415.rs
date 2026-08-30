use std::io::{Read, Write};
use std::collections::HashMap;

fn parse_addr(s: &str) -> Option<(bool, u128)> {
    if s.contains(':') {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 8 { return None; }
        let mut v: u128 = 0;
        for p in parts {
            if p.is_empty() || p.len() > 4 { return None; }
            let mut g: u128 = 0;
            for c in p.chars() {
                let d = c.to_digit(16)?;
                g = g * 16 + d as u128;
            }
            v = (v << 16) | g;
        }
        Some((false, v))
    } else {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 { return None; }
        let mut v: u128 = 0;
        for p in parts {
            if p.is_empty() || p.len() > 3 { return None; }
            let mut g: u128 = 0;
            for c in p.chars() {
                let d = c.to_digit(10)?;
                g = g * 10 + d as u128;
            }
            if g > 255 { return None; }
            v = (v << 8) | g;
        }
        Some((true, v))
    }
}

fn mask(bits: u32, total: u32) -> u128 {
    if bits == 0 { 0 } else { (!0u128) << (total - bits) >> (128 - total) << 0 }
}

fn prefix_of(v: u128, bits: u32, total: u32) -> u128 {
    if bits == 0 { return 0; }
    if bits >= total { return v; }
    v >> (total - bits) << (total - bits)
}

struct Dsu { p: Vec<usize>, r: Vec<u32>, e: Vec<i128>, t: Vec<i128> }

impl Dsu {
    fn find(&mut self, mut x: usize) -> usize {
        while self.p[x] != x {
            self.p[x] = self.p[self.p[x]];
            x = self.p[x];
        }
        x
    }
}

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_ascii_whitespace();
    let out = std::io::stdout();
    let mut w = std::io::BufWriter::new(out.lock());

    let a: i128 = it.next().unwrap().parse().unwrap();
    let pp: i128 = it.next().unwrap().parse().unwrap();
    let b: i128 = it.next().unwrap().parse().unwrap();
    let g = it.next().unwrap().to_string();
    let k: i128 = it.next().unwrap().parse().unwrap();
    let d: i128 = it.next().unwrap().parse().unwrap();
    let e: usize = it.next().unwrap().parse().unwrap();

    let mut cidrs: Vec<(bool, u128, i128)> = Vec::with_capacity(e);
    for _ in 0..e {
        let tok = it.next().unwrap();
        let pos = tok.rfind('/').unwrap();
        let addr = &tok[..pos];
        let pre: i128 = tok[pos + 1..].parse().unwrap();
        let pa = parse_addr(addr);
        match pa {
            Some((v4, v)) => cidrs.push((v4, v, pre)),
            None => cidrs.push((true, 0, -1)),
        }
    }

    let mut invalid = false;
    if d != -1 && d < 1 { invalid = true; }
    if g == "DEFAULT" {
    } else if g == "V6" {
        if k < 1 || k > 128 { invalid = true; }
    } else {
        invalid = true;
    }
    if d == -1 && !invalid {
        for c in cidrs.iter() {
            let lim = if c.0 { 32 } else { 128 };
            if c.2 < 0 || c.2 > lim { invalid = true; break; }
        }
    }

    if invalid {
        writeln!(w, "INVALID").unwrap();
        return;
    }

    let norm_cidrs: Vec<(bool, u128, u32)> = cidrs
        .iter()
        .map(|c| {
            let total = if c.0 { 32 } else { 128 };
            let bits = c.2 as u32;
            (c.0, prefix_of(c.1, bits, total), bits)
        })
        .collect();

    let kk = if g == "V6" { k as u32 } else { 128 };
    let default_mode = g == "DEFAULT";

    let cap: i128 = b * pp;

    let q: usize = it.next().unwrap().parse().unwrap();
    let mut map: HashMap<(bool, u128), usize> = HashMap::new();
    let mut dsu = Dsu { p: Vec::new(), r: Vec::new(), e: Vec::new(), t: Vec::new() };

    let mut res = String::new();

    for _ in 0..q {
        let ty = it.next().unwrap();
        if ty == "R" {
            let t: i128 = it.next().unwrap().parse().unwrap();
            let c: i128 = it.next().unwrap().parse().unwrap();
            let peer = it.next().unwrap().to_string();
            let n: usize = it.next().unwrap().parse().unwrap();
            let mut addrs: Vec<&str> = Vec::with_capacity(n);
            for _ in 0..n { addrs.push(it.next().unwrap()); }
            let chosen: String;
            if d >= 1 {
                let di = d;
                if di <= n as i128 {
                    chosen = addrs[n - di as usize].to_string();
                } else {
                    chosen = peer.clone();
                }
            } else {
                let mut pick: Option<String> = None;
                for i in (0..n).rev() {
                    let pa = parse_addr(addrs[i]);
                    if let Some((v4, v)) = pa {
                        let total = if v4 { 32u32 } else { 128u32 };
                        let mut excluded = false;
                        for cd in norm_cidrs.iter() {
                            if cd.0 != v4 { continue; }
                            if prefix_of(v, cd.2, total) == cd.1 { excluded = true; break; }
                        }
                        if !excluded { pick = Some(addrs[i].to_string()); break; }
                    } else {
                        pick = Some(addrs[i].to_string());
                        break;
                    }
                }
                chosen = match pick { Some(x) => x, None => peer.clone() };
            }
            let key = match parse_addr(&chosen) {
                Some((v4, v)) => {
                    if v4 || default_mode { (v4, v) } else { (false, prefix_of(v, kk, 128)) }
                }
                None => (true, 0),
            };
            let id = *map.entry(key).or_insert_with(|| {
                dsu.p.push(dsu.p.len());
                dsu.r.push(0);
                dsu.e.push(cap);
                dsu.t.push(t);
                dsu.p.len() - 1
            });
            let r = dsu.find(id);
            let dt = t - dsu.t[r];
            if dt > 0 {
                let gain = dt.saturating_mul(a);
                let ne = dsu.e[r].saturating_add(gain);
                dsu.e[r] = if ne > cap { cap } else { ne };
            }
            dsu.t[r] = t;
            let cost = c * pp;
            if dsu.e[r] >= cost {
                dsu.e[r] -= cost;
                res.push('1');
            } else {
                res.push('0');
            }
            res.push('\n');
        } else {
            let t: i128 = it.next().unwrap().parse().unwrap();
            let i1 = it.next().unwrap().to_string();
            let i2 = it.next().unwrap().to_string();
            let mut ids = [0usize; 2];
            for (j, sx) in [i1, i2].iter().enumerate() {
                let key = match parse_addr(sx) {
                    Some((v4, v)) => {
                        if v4 || default_mode { (v4, v) } else { (false, prefix_of(v, kk, 128)) }
                    }
                    None => (true, 0),
                };
                let id = *map.entry(key).or_insert_with(|| {
                    dsu.p.push(dsu.p.len());
                    dsu.r.push(0);
                    dsu.e.push(cap);
                    dsu.t.push(t);
                    dsu.p.len() - 1
                });
                ids[j] = id;
            }
            let r1 = dsu.find(ids[0]);
            let r2 = dsu.find(ids[1]);
            let dt1 = t - dsu.t[r1];
            if dt1 > 0 {
                let gain = dt1.saturating_mul(a);
                let ne = dsu.e[r1].saturating_add(gain);
                dsu.e[r1] = if ne > cap { cap } else { ne };
            }
            dsu.t[r1] = t;
            if r1 != r2 {
                let dt2 = t - dsu.t[r2];
                if dt2 > 0 {
                    let gain = dt2.saturating_mul(a);
                    let ne = dsu.e[r2].saturating_add(gain);
                    dsu.e[r2] = if ne > cap { cap } else { ne };
                }
                dsu.t[r2] = t;
                let sum = dsu.e[r1].saturating_add(dsu.e[r2]);
                let ne = if sum > cap { cap } else { sum };
                let (big, small) = if dsu.r[r1] >= dsu.r[r2] { (r1, r2) } else { (r2, r1) };
                dsu.p[small] = big;
                if dsu.r[big] == dsu.r[small] { dsu.r[big] += 1; }
                dsu.e[big] = ne;
                dsu.t[big] = t;
            }
        }
    }
    let _ = mask(0, 32);
    w.write_all(res.as_bytes()).unwrap();
}