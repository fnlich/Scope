use std::io::{self, Read, Write};
use std::collections::HashMap;

fn main() {
    let mut inp = String::new();
    io::stdin().read_to_string(&mut inp).unwrap();
    let mut it = inp.split_ascii_whitespace();
    let n: usize = match it.next() { Some(x) => x.parse().unwrap(), None => 0 };
    let mut ids: Vec<u64> = Vec::with_capacity(n);
    let mut par: Vec<u64> = Vec::with_capacity(n);
    let mut ns: Vec<&str> = Vec::with_capacity(n);
    let mut nm: Vec<&str> = Vec::with_capacity(n);
    let mut vis: Vec<u8> = Vec::with_capacity(n);
    let mut st: Vec<u8> = Vec::with_capacity(n);
    for _ in 0..n {
        let a: u64 = it.next().unwrap().parse().unwrap();
        let b: u64 = it.next().unwrap().parse().unwrap();
        let c = it.next().unwrap();
        let d = it.next().unwrap();
        let e = it.next().unwrap().as_bytes()[0];
        let f = it.next().unwrap().as_bytes()[0];
        ids.push(a); par.push(b); ns.push(c); nm.push(d); vis.push(e); st.push(f);
    }
    let mut idx: HashMap<u64, usize> = HashMap::with_capacity(n * 2);
    for i in 0..n { idx.insert(ids[i], i); }
    let mut pidx: Vec<i64> = vec![-1; n];
    let mut local_bad: Vec<bool> = vec![false; n];
    for i in 0..n {
        if par[i] == 0 {
            if vis[i] == b'N' { local_bad[i] = true; }
        } else {
            if vis[i] == b'P' { local_bad[i] = true; }
            if ns[i] != "-" { local_bad[i] = true; }
            match idx.get(&par[i]) {
                Some(&p) => { pidx[i] = p as i64; }
                None => { local_bad[i] = true; }
            }
        }
    }
    // 0 unknown, 1 in progress, 2 done
    let mut state: Vec<u8> = vec![0; n];
    let mut bad: Vec<bool> = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for s in 0..n {
        if state[s] == 2 { continue; }
        let mut cur = s;
        loop {
            if state[cur] == 2 { break; }
            if state[cur] == 1 {
                // cycle: mark all in progress on stack as bad
                bad[cur] = true;
                break;
            }
            state[cur] = 1;
            stack.push(cur);
            if local_bad[cur] || pidx[cur] < 0 {
                bad[cur] = true;
                break;
            }
            cur = pidx[cur] as usize;
        }
        let base_bad = if state[cur] == 2 { bad[cur] } else { true };
        let mut carry = base_bad;
        while let Some(x) = stack.pop() {
            if state[x] == 2 { continue; }
            if x == cur {
                bad[x] = true;
                state[x] = 2;
                carry = true;
                continue;
            }
            let b = bad[x] || carry;
            bad[x] = b;
            state[x] = 2;
            carry = b;
        }
    }
    let mut visible: Vec<bool> = vec![false; n];
    let mut canon_start: Vec<usize> = vec![0; n];
    let mut canon_len: Vec<usize> = vec![0; n];
    let mut buf: Vec<u8> = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();
    let mut depth: Vec<u32> = vec![0; n];
    for i in 0..n {
        let mut d = 0u32;
        let mut c = i as i64;
        let mut steps = 0;
        while c >= 0 && !bad[c as usize] {
            let p = pidx[c as usize];
            if p < 0 { break; }
            d += 1; c = p;
            steps += 1;
            if steps > 300001 { break; }
        }
        depth[i] = d;
    }
    order.sort_by_key(|&i| depth[i]);
    for &i in order.iter() {
        if bad[i] { continue; }
        if par[i] == 0 {
            if vis[i] == b'P' {
                visible[i] = true;
                let s = buf.len();
                if ns[i] != "-" { buf.extend_from_slice(ns[i].as_bytes()); buf.push(b'.'); }
                buf.extend_from_slice(nm[i].as_bytes());
                canon_start[i] = s; canon_len[i] = buf.len() - s;
            }
        } else {
            let p = pidx[i] as usize;
            if visible[p] && vis[i] == b'N' {
                visible[i] = true;
                let s = buf.len();
                let (ps, pl) = (canon_start[p], canon_len[p]);
                for k in 0..pl { let b = buf[ps + k]; buf.push(b); }
                buf.push(b'+');
                buf.extend_from_slice(nm[i].as_bytes());
                canon_start[i] = s; canon_len[i] = buf.len() - s;
            }
        }
    }
    let mut cat: HashMap<Vec<u8>, (usize, bool)> = HashMap::new();
    let out = io::stdout();
    let mut w = io::BufWriter::with_capacity(1 << 20, out.lock());
    for i in 0..n {
        if bad[i] { w.write_all(b"INVALID\n").unwrap(); continue; }
        if !visible[i] { w.write_all(b"HIDDEN\n").unwrap(); continue; }
        let s = canon_start[i]; let l = canon_len[i];
        let mut key: Vec<u8> = Vec::with_capacity(l);
        for k in 0..l { key.push(buf[s + k].to_ascii_lowercase()); }
        let cur = st[i] == b'C';
        let action: &[u8];
        match cat.get(&key) {
            None => { cat.insert(key, (i, cur)); action = b"INSERTED "; }
            Some(&(_, oldcur)) => {
                if !oldcur && cur { cat.insert(key, (i, cur)); action = b"REPLACED "; }
                else { action = b"IGNORED "; }
            }
        }
        w.write_all(action).unwrap();
        w.write_all(&buf[s..s + l]).unwrap();
        w.write_all(b"\n").unwrap();
    }
}