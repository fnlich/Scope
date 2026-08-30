use std::io::{self, Read, Write};
use std::collections::HashMap;

fn tok(d: &[u8], p: &mut usize) -> (usize, usize) {
    while *p < d.len() && (d[*p] == b' ' || d[*p] == b'\n' || d[*p] == b'\r' || d[*p] == b'\t') {
        *p += 1;
    }
    let s = *p;
    while *p < d.len() && !(d[*p] == b' ' || d[*p] == b'\n' || d[*p] == b'\r' || d[*p] == b'\t') {
        *p += 1;
    }
    (s, *p)
}

fn parse_u64(d: &[u8], s: usize, e: usize) -> u64 {
    let mut v: u64 = 0;
    for i in s..e {
        v = v * 10 + (d[i] - b'0') as u64;
    }
    v
}

fn go(cm: &HashMap<u64, u32>, fail: &[u32], mut u: u32, c: u8) -> u32 {
    loop {
        if let Some(&v) = cm.get(&((u as u64) * 26 + c as u64)) {
            return v;
        }
        if u == 0 {
            return 0;
        }
        u = fail[u as usize];
    }
}

fn tsum(trackpats: &Vec<Vec<u32>>, tpar: &[u8], tempty: u8, x: usize) -> u8 {
    let mut r = tempty;
    for &pid in trackpats[x].iter() {
        r ^= tpar[pid as usize];
    }
    r
}

fn main() {
    let mut data = Vec::new();
    io::stdin().read_to_end(&mut data).unwrap();
    let mut pos = 0usize;

    let (s, e) = tok(&data, &mut pos);
    let n = parse_u64(&data, s, e) as usize;
    let (s, e) = tok(&data, &mut pos);
    let q = parse_u64(&data, s, e) as usize;

    let mut labels: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut sizes: Vec<u64> = Vec::with_capacity(n);
    let mut sel: Vec<u8> = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b) = tok(&data, &mut pos);
        labels.push((a, b));
        let (a2, b2) = tok(&data, &mut pos);
        sizes.push(parse_u64(&data, a2, b2));
        let (a3, b3) = tok(&data, &mut pos);
        sel.push(parse_u64(&data, a3, b3) as u8);
    }

    let mut pat_map: HashMap<&[u8], u32> = HashMap::new();
    let mut pat_list: Vec<(usize, usize)> = Vec::new();
    let mut cmds: Vec<(u8, u32)> = Vec::with_capacity(q);
    let mut cur: u32 = u32::MAX;

    for _ in 0..q {
        let (a, b) = tok(&data, &mut pos);
        if b <= a {
            break;
        }
        let ch = data[a];
        if ch == b'F' {
            let (pa, pb) = tok(&data, &mut pos);
            let key = &data[pa..pb];
            if let Some(&id) = pat_map.get(key) {
                cur = id;
            } else {
                let id = pat_list.len() as u32;
                pat_list.push((pa, pb));
                pat_map.insert(key, id);
                cur = id;
            }
        } else if ch == b'C' {
            cur = u32::MAX;
        } else if ch == b'S' {
            cmds.push((0, cur));
        } else if ch == b'D' {
            cmds.push((1, cur));
        } else if ch == b'T' {
            cmds.push((2, cur));
        }
    }

    let p = pat_list.len();

    let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
    let mut term: Vec<i32> = vec![-1];
    let mut child_map: HashMap<u64, u32> = HashMap::new();

    for id in 0..p {
        let (ps, pe) = pat_list[id];
        let mut c0: u32 = 0;
        for i in ps..pe {
            let c = data[i] - b'a';
            let k = (c0 as u64) * 26 + c as u64;
            if let Some(&v) = child_map.get(&k) {
                c0 = v;
            } else {
                let v = children.len() as u32;
                children.push(Vec::new());
                term.push(-1);
                child_map.insert(k, v);
                children[c0 as usize].push((c, v));
                c0 = v;
            }
        }
        term[c0 as usize] = id as i32;
    }

    let nodes = children.len();
    let mut fail: Vec<u32> = vec![0; nodes];
    let mut dict: Vec<u32> = vec![0; nodes];
    let mut queue: Vec<u32> = Vec::with_capacity(nodes);

    for idx in 0..children[0].len() {
        let (_, v) = children[0][idx];
        fail[v as usize] = 0;
        dict[v as usize] = 0;
        queue.push(v);
    }
    let mut head = 0usize;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        let deg = children[u as usize].len();
        for idx in 0..deg {
            let (c, v) = children[u as usize][idx];
            let f = go(&child_map, &fail, fail[u as usize], c);
            fail[v as usize] = f;
            dict[v as usize] = if term[f as usize] >= 0 {
                f
            } else {
                dict[f as usize]
            };
            queue.push(v);
        }
    }

    let mut cnt: Vec<u32> = vec![0; p];
    let mut touched: Vec<u32> = Vec::new();
    let mut matches: Vec<Vec<u32>> = vec![Vec::new(); p];
    let mut trackpats: Vec<Vec<u32>> = vec![Vec::new(); n];

    for i in 0..n {
        let (ls, le) = labels[i];
        let mut node: u32 = 0;
        for j in ls..le {
            let c = data[j] - b'a';
            node = go(&child_map, &fail, node, c);
            let mut jn = if term[node as usize] >= 0 {
                node
            } else {
                dict[node as usize]
            };
            while jn != 0 {
                let pid = term[jn as usize] as usize;
                if cnt[pid] == 0 {
                    touched.push(pid as u32);
                }
                cnt[pid] += 1;
                jn = dict[jn as usize];
            }
        }
        for k in 0..touched.len() {
            let pid = touched[k] as usize;
            matches[pid].push(i as u32);
            if cnt[pid] & 1 == 1 {
                trackpats[i].push(pid as u32);
            }
            cnt[pid] = 0;
        }
        touched.clear();
    }

    let mut finalized: Vec<bool> = vec![false; n];
    let mut value: Vec<u8> = vec![0; n];
    let mut nfin = 0usize;
    let mut closed: Vec<bool> = vec![false; p];
    let mut tpar: Vec<u8> = vec![0; p];
    let mut tempty: u8 = 0;

    let mut i = cmds.len();
    while i > 0 {
        i -= 1;
        if nfin == n {
            break;
        }
        let (t, pid) = cmds[i];
        if t == 2 {
            if pid == u32::MAX {
                tempty ^= 1;
            } else {
                tpar[pid as usize] ^= 1;
            }
            continue;
        }
        let val: u8 = if t == 0 { 1 } else { 0 };
        if pid == u32::MAX {
            for x in 0..n {
                if !finalized[x] {
                    finalized[x] = true;
                    nfin += 1;
                    value[x] = val ^ tsum(&trackpats, &tpar, tempty, x);
                }
            }
        } else {
            let pi = pid as usize;
            if closed[pi] {
                continue;
            }
            closed[pi] = true;
            for k in 0..matches[pi].len() {
                let x = matches[pi][k] as usize;
                if !finalized[x] {
                    finalized[x] = true;
                    nfin += 1;
                    value[x] = val ^ tsum(&trackpats, &tpar, tempty, x);
                }
            }
        }
    }

    for x in 0..n {
        if !finalized[x] {
            value[x] = sel[x] ^ tsum(&trackpats, &tpar, tempty, x);
        }
    }

    let mut total: u64 = 0;
    for x in 0..n {
        if value[x] == 1 {
            total += sizes[x];
        }
    }

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut buf = String::with_capacity(n * 2 + 32);
    buf.push_str(&total.to_string());
    buf.push('\n');
    for x in 0..n {
        if x > 0 {
            buf.push(' ');
        }
        buf.push(if value[x] == 1 { '1' } else { '0' });
    }
    buf.push('\n');
    out.write_all(buf.as_bytes()).unwrap();
}