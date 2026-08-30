use std::io::{self, Read, Write};

fn read_int(d: &[u8], p: &mut usize) -> i64 {
    while *p < d.len() && (d[*p] as char).is_whitespace() {
        *p += 1;
    }
    let mut neg = false;
    if *p < d.len() && (d[*p] == b'-' || d[*p] == b'+') {
        neg = d[*p] == b'-';
        *p += 1;
    }
    let mut v: i64 = 0;
    while *p < d.len() && d[*p] >= b'0' && d[*p] <= b'9' {
        v = v * 10 + (d[*p] - b'0') as i64;
        *p += 1;
    }
    if neg { -v } else { v }
}

fn read_op(d: &[u8], p: &mut usize) -> u8 {
    while *p < d.len() && (d[*p] as char).is_whitespace() {
        *p += 1;
    }
    if *p < d.len() {
        let c = d[*p];
        *p += 1;
        c
    } else {
        0
    }
}

fn rebuild(
    n: usize,
    nxt: &[u32],
    missing: &[bool],
    perm: &[bool],
    res: &mut Vec<u32>,
    cyc: &mut Vec<i32>,
    compcyc: &mut Vec<i32>,
    oncyc: &mut Vec<bool>,
    tin: &mut Vec<u32>,
    tout: &mut Vec<u32>,
) {
    for i in 0..=n {
        res[i] = 0;
        cyc[i] = -1;
        compcyc[i] = -1;
        oncyc[i] = false;
        tin[i] = 0;
        tout[i] = 0;
    }
    let mut color = vec![0u8; n + 1];
    let mut pos = vec![0u32; n + 1];
    let mut path: Vec<u32> = Vec::with_capacity(64);
    let mut ncyc: i32 = 0;
    for i in 1..=n {
        if color[i] != 0 {
            continue;
        }
        path.clear();
        let mut x = i as u32;
        loop {
            if color[x as usize] == 1 {
                let st = pos[x as usize] as usize;
                let id = ncyc;
                ncyc += 1;
                for k in st..path.len() {
                    let v = path[k] as usize;
                    oncyc[v] = true;
                    cyc[v] = id;
                }
                break;
            }
            if color[x as usize] == 2 {
                break;
            }
            color[x as usize] = 1;
            pos[x as usize] = path.len() as u32;
            path.push(x);
            let nx = nxt[x as usize];
            if nx == 0 {
                break;
            }
            x = nx;
        }
        for &v in path.iter() {
            color[v as usize] = 2;
        }
    }

    let mut minperm = vec![u32::MAX; ncyc.max(1) as usize];
    for v in 1..=n {
        if oncyc[v] && perm[v] {
            let id = cyc[v] as usize;
            if (v as u32) < minperm[id] {
                minperm[id] = v as u32;
            }
        }
    }

    let mut done = vec![false; n + 1];
    for v in 1..=n {
        if oncyc[v] {
            let id = cyc[v] as usize;
            res[v] = if minperm[id] == u32::MAX { 0 } else { minperm[id] };
            compcyc[v] = cyc[v];
            done[v] = true;
        } else if nxt[v] == 0 {
            res[v] = if !missing[v] && perm[v] { v as u32 } else { 0 };
            compcyc[v] = -1;
            done[v] = true;
        }
    }

    let mut stack2: Vec<u32> = Vec::with_capacity(64);
    for i in 1..=n {
        if done[i] {
            continue;
        }
        stack2.clear();
        let mut x = i as u32;
        while !done[x as usize] {
            stack2.push(x);
            done[x as usize] = true;
            x = nxt[x as usize];
        }
        for k in (0..stack2.len()).rev() {
            let v = stack2[k] as usize;
            let nv = nxt[v] as usize;
            res[v] = res[nv];
            compcyc[v] = compcyc[nv];
        }
    }

    let mut head = vec![0u32; n + 1];
    let mut sib = vec![0u32; n + 1];
    for x in 1..=n {
        if !oncyc[x] && nxt[x] != 0 {
            let p = nxt[x] as usize;
            sib[x] = head[p];
            head[p] = x as u32;
        }
    }
    let mut it = head.clone();
    let mut timer: u32 = 1;
    let mut st: Vec<u32> = Vec::with_capacity(64);
    for r in 1..=n {
        if oncyc[r] || nxt[r] == 0 {
            st.clear();
            st.push(r as u32);
            tin[r] = timer;
            timer += 1;
            while let Some(&top) = st.last() {
                let t = top as usize;
                if it[t] != 0 {
                    let c = it[t];
                    it[t] = sib[c as usize];
                    tin[c as usize] = timer;
                    timer += 1;
                    st.push(c);
                } else {
                    tout[t] = timer;
                    timer += 1;
                    st.pop();
                }
            }
        }
    }
}

fn main() {
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf).unwrap();
    let d = &buf[..];
    let mut p = 0usize;
    let n = read_int(d, &mut p) as usize;
    let q = read_int(d, &mut p) as usize;

    let mut nxt = vec![0u32; n + 1];
    let mut missing = vec![false; n + 1];
    let mut perm = vec![false; n + 1];
    for i in 1..=n {
        let t = read_int(d, &mut p);
        let r = read_int(d, &mut p);
        perm[i] = t == 0;
        if r >= 1 && r <= n as i64 {
            nxt[i] = r as u32;
            missing[i] = false;
        } else {
            nxt[i] = 0;
            missing[i] = r > n as i64;
        }
    }

    let mut res = vec![0u32; n + 1];
    let mut cyc = vec![-1i32; n + 1];
    let mut compcyc = vec![-1i32; n + 1];
    let mut oncyc = vec![false; n + 1];
    let mut tin = vec![0u32; n + 1];
    let mut tout = vec![0u32; n + 1];
    let mut dirty = true;

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for _ in 0..q {
        let op = read_op(d, &mut p);
        if op == b'U' {
            let i = read_int(d, &mut p) as usize;
            let r = read_int(d, &mut p);
            if r >= 1 && r <= n as i64 {
                nxt[i] = r as u32;
                missing[i] = false;
            } else {
                nxt[i] = 0;
                missing[i] = r > n as i64;
            }
            dirty = true;
        } else if op == b'C' {
            let o = read_int(d, &mut p);
            let s = read_int(d, &mut p);
            if dirty {
                rebuild(
                    n, &nxt, &missing, &perm, &mut res, &mut cyc, &mut compcyc, &mut oncyc,
                    &mut tin, &mut tout,
                );
                dirty = false;
            }
            let o_ok = o >= 1 && o <= n as i64;
            let oi = if o_ok { o as usize } else { 0 };
            let s_ok = s >= 1 && s <= n as i64 && perm[s as usize];
            let si = if s >= 1 && s <= n as i64 { s as usize } else { 0 };

            if !s_ok {
                if o_ok && res[oi] != 0 {
                    writeln!(out, "O {}", res[oi]).unwrap();
                } else {
                    writeln!(out, "NONE 0").unwrap();
                }
            } else {
                let mut reversed = false;
                if o_ok && oi != si {
                    let inw = if oncyc[si] {
                        oncyc[oi] && cyc[oi] == cyc[si]
                    } else {
                        (tin[oi] <= tin[si] && tout[si] <= tout[oi])
                            || (oncyc[oi] && compcyc[si] >= 0 && cyc[oi] == compcyc[si])
                    };
                    reversed = inw;
                }
                if reversed {
                    if res[oi] != 0 {
                        writeln!(out, "O {}", res[oi]).unwrap();
                    } else if res[si] != 0 {
                        writeln!(out, "S {}", res[si]).unwrap();
                    } else {
                        writeln!(out, "NONE 0").unwrap();
                    }
                } else {
                    if res[si] != 0 {
                        writeln!(out, "S {}", res[si]).unwrap();
                    } else if o_ok && res[oi] != 0 {
                        writeln!(out, "O {}", res[oi]).unwrap();
                    } else {
                        writeln!(out, "NONE 0").unwrap();
                    }
                }
            }
        }
    }
}