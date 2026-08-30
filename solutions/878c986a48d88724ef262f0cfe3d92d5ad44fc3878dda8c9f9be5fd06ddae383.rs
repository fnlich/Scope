use std::io::{self, Read, Write};

fn main() {
    let mut inp = String::new();
    io::stdin().read_to_string(&mut inp).unwrap();
    let mut it = inp.split_ascii_whitespace();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let n: usize = it.next().unwrap().parse().unwrap();
    let m: usize = it.next().unwrap().parse().unwrap();
    let s: usize = it.next().unwrap().parse().unwrap();

    let mut attached = vec![false; n + 1];
    let mut valid = vec![false; n + 1];
    let mut req_start = vec![0u32; n + 2];
    let mut req_flat: Vec<u32> = Vec::new();
    for i in 1..=n {
        let a: u32 = it.next().unwrap().parse().unwrap();
        let g: u32 = it.next().unwrap().parse().unwrap();
        let r: usize = it.next().unwrap().parse().unwrap();
        attached[i] = a == 1;
        valid[i] = g == 1;
        req_start[i] = req_flat.len() as u32;
        for _ in 0..r {
            let x: u32 = it.next().unwrap().parse().unwrap();
            req_flat.push(x);
        }
        req_start[i + 1] = req_flat.len() as u32;
    }

    let mut eu: Vec<u32> = Vec::with_capacity(m);
    let mut ev: Vec<u32> = Vec::with_capacity(m);
    let mut deg = vec![0u32; n + 2];
    for _ in 0..m {
        let u: u32 = it.next().unwrap().parse().unwrap();
        let v: u32 = it.next().unwrap().parse().unwrap();
        eu.push(u);
        ev.push(v);
        deg[u as usize] += 1;
    }
    let mut estart = vec![0u32; n + 2];
    let mut acc = 0u32;
    for i in 1..=n {
        estart[i] = acc;
        acc += deg[i];
    }
    estart[n + 1] = acc;
    let mut fill = estart.clone();
    let mut eflat = vec![0u32; m];
    for i in 0..m {
        let u = eu[i] as usize;
        eflat[fill[u] as usize] = ev[i];
        fill[u] += 1;
    }

    let mut bind: Vec<u32> = vec![0u32; s + 2];
    let mut state = vec![0u8; n + 1];
    for i in 1..=n {
        if attached[i] {
            state[i] = 1;
        }
    }

    let mut stamp = vec![0u32; n + 1];
    let mut lid = vec![0u32; n + 1];
    let mut cur: u32 = 0;

    let q: usize = it.next().unwrap().parse().unwrap();
    for _ in 0..q {
        let cmd = it.next().unwrap();
        if cmd == "D" {
            let x: usize = it.next().unwrap().parse().unwrap();
            let y: u32 = it.next().unwrap().parse().unwrap();
            bind[x] = y;
        } else if cmd == "U" {
            let x: usize = it.next().unwrap().parse().unwrap();
            bind[x] = 0;
        } else {
            let x: usize = it.next().unwrap().parse().unwrap();
            let f: u32 = it.next().unwrap().parse().unwrap();
            let p = it.next().unwrap();
            let pol_d = p == "D";

            if attached[x] || (f == 0 && state[x] == 1) {
                out.write_all(b"S C\n").unwrap();
                continue;
            }

            cur += 1;
            let mut examined: Vec<u32> = Vec::new();
            let mut adj_range: Vec<(u32, u32)> = Vec::new();
            let mut bad: Vec<bool> = Vec::new();
            let mut flat: Vec<u32> = Vec::new();
            let mut work: Vec<u32> = Vec::new();

            stamp[x] = cur;
            lid[x] = 0;
            examined.push(x as u32);
            adj_range.push((0, 0));
            bad.push(false);
            work.push(x as u32);

            while let Some(u) = work.pop() {
                let uu = u as usize;
                let lu = lid[uu] as usize;
                let st = flat.len() as u32;
                let mut b = !valid[uu];
                let a0 = estart[uu] as usize;
                let a1 = estart[uu + 1] as usize;
                for i in a0..a1 {
                    let t = eflat[i] as usize;
                    if attached[t] {
                        continue;
                    }
                    if f == 0 && state[t] == 1 {
                        continue;
                    }
                    if stamp[t] != cur {
                        stamp[t] = cur;
                        lid[t] = examined.len() as u32;
                        examined.push(t as u32);
                        adj_range.push((0, 0));
                        bad.push(false);
                        work.push(t as u32);
                    }
                    flat.push(lid[t]);
                }
                let r0 = req_start[uu] as usize;
                let r1 = req_start[uu + 1] as usize;
                for i in r0..r1 {
                    let nm = req_flat[i] as usize;
                    let tb = bind[nm];
                    if tb == 0 {
                        b = true;
                        continue;
                    }
                    let t = tb as usize;
                    if attached[t] {
                        continue;
                    }
                    if f == 0 && state[t] == 1 {
                        continue;
                    }
                    if stamp[t] != cur {
                        stamp[t] = cur;
                        lid[t] = examined.len() as u32;
                        examined.push(t as u32);
                        adj_range.push((0, 0));
                        bad.push(false);
                        work.push(t as u32);
                    }
                    flat.push(lid[t]);
                }
                adj_range[lu] = (st, flat.len() as u32);
                bad[lu] = b;
            }

            let k = examined.len();
            let mut index = vec![u32::MAX; k];
            let mut low = vec![0u32; k];
            let mut onstk = vec![false; k];
            let mut comp = vec![u32::MAX; k];
            let mut sstack: Vec<u32> = Vec::new();
            let mut comp_ok: Vec<bool> = Vec::new();
            let mut counter: u32 = 0;
            let mut call: Vec<(u32, u32)> = Vec::new();

            for start in 0..k {
                if index[start] != u32::MAX {
                    continue;
                }
                index[start] = counter;
                low[start] = counter;
                counter += 1;
                sstack.push(start as u32);
                onstk[start] = true;
                call.push((start as u32, adj_range[start].0));
                while !call.is_empty() {
                    let last = call.len() - 1;
                    let (v, ep) = call[last];
                    let vi = v as usize;
                    if ep < adj_range[vi].1 {
                        call[last].1 = ep + 1;
                        let w = flat[ep as usize] as usize;
                        if index[w] == u32::MAX {
                            index[w] = counter;
                            low[w] = counter;
                            counter += 1;
                            sstack.push(w as u32);
                            onstk[w] = true;
                            call.push((w as u32, adj_range[w].0));
                        } else if onstk[w] {
                            if index[w] < low[vi] {
                                low[vi] = index[w];
                            }
                        }
                    } else {
                        call.pop();
                        if low[vi] == index[vi] {
                            let cid = comp_ok.len() as u32;
                            let mut members: Vec<u32> = Vec::new();
                            loop {
                                let w = sstack.pop().unwrap();
                                onstk[w as usize] = false;
                                comp[w as usize] = cid;
                                members.push(w);
                                if w == v {
                                    break;
                                }
                            }
                            let mut ok = true;
                            for &mm in members.iter() {
                                if bad[mm as usize] {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                'outer: for &mm in members.iter() {
                                    let (a, b2) = adj_range[mm as usize];
                                    for i in a..b2 {
                                        let w = flat[i as usize] as usize;
                                        let cw = comp[w];
                                        if cw != cid {
                                            if !comp_ok[cw as usize] {
                                                ok = false;
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                            }
                            comp_ok.push(ok);
                        }
                        if let Some(&(pv, _)) = call.last() {
                            if low[vi] < low[pv as usize] {
                                low[pv as usize] = low[vi];
                            }
                        }
                    }
                }
            }

            let ok0 = comp_ok[comp[0] as usize];
            if ok0 {
                for i in 0..k {
                    state[examined[i] as usize] = 1;
                }
                out.write_all(b"S C\n").unwrap();
            } else if pol_d {
                for i in 0..k {
                    let nd = examined[i] as usize;
                    state[nd] = if comp_ok[comp[i] as usize] { 1 } else { 2 };
                }
                let c = match state[x] {
                    0 => "U",
                    1 => "C",
                    _ => "P",
                };
                out.write_all(b"D ").unwrap();
                out.write_all(c.as_bytes()).unwrap();
                out.write_all(b"\n").unwrap();
            } else {
                let c = match state[x] {
                    0 => "U",
                    1 => "C",
                    _ => "P",
                };
                out.write_all(b"E ").unwrap();
                out.write_all(c.as_bytes()).unwrap();
                out.write_all(b"\n").unwrap();
            }
        }
    }
}