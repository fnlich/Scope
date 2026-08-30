use std::io::{self, Read, Write};

const LOG: usize = 20;

fn main() {
    let mut inp = String::new();
    io::stdin().read_to_string(&mut inp).unwrap();
    let mut it = inp.split_ascii_whitespace().map(|t| t.parse::<u64>().unwrap());

    let n = it.next().unwrap() as usize;
    let q = it.next().unwrap() as usize;
    let h0 = it.next().unwrap();
    let s0 = it.next().unwrap();
    let r0 = it.next().unwrap();
    let _ = s0;

    let cap = n + q + 2;
    let mut owner: Vec<u64> = Vec::with_capacity(cap);
    let mut seqv: Vec<u64> = Vec::with_capacity(cap);
    let mut up: Vec<Vec<u32>> = vec![Vec::with_capacity(cap); LOG];

    let mut pw: Vec<u64> = Vec::with_capacity(n);
    let mut ps: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        pw.push(it.next().unwrap());
        ps.push(it.next().unwrap());
    }

    let mut prev: Option<u32> = None;
    for i in (0..n).rev() {
        let id = owner.len() as u32;
        owner.push(pw[i]);
        seqv.push(ps[i]);
        let p0 = match prev {
            Some(p) => p,
            None => id,
        };
        up[0].push(p0);
        for j in 1..LOG {
            let mid = up[j - 1][id as usize];
            let a = up[j - 1][mid as usize];
            up[j].push(a);
        }
        prev = Some(id);
    }
    let root_node = prev.unwrap();

    let mut vnode: Vec<u32> = Vec::with_capacity(q + 1);
    let mut vlen: Vec<u32> = Vec::with_capacity(q + 1);
    let mut vh: Vec<u64> = Vec::with_capacity(q + 1);
    let mut vr: Vec<u64> = Vec::with_capacity(q + 1);
    vnode.push(root_node);
    vlen.push(n as u32);
    vh.push(h0);
    vr.push(r0);

    let mut out = String::with_capacity(q * 40 + 16);

    for _ in 0..q {
        let p = it.next().unwrap() as usize;
        let x = it.next().unwrap();
        let s = it.next().unwrap();
        let r = it.next().unwrap();
        let k = it.next().unwrap();

        let hp = vh[p];
        let pnode = vnode[p];
        let plen = vlen[p];
        let rp = vr[p];

        let nh;
        let mut nn: u32;
        let mut nl: u32;

        if s < hp {
            nh = hp;
            nn = pnode;
            nl = plen;
        } else {
            nh = s;
            if owner[pnode as usize] == x {
                nn = pnode;
                nl = plen;
            } else {
                let mut base = pnode;
                let mut blen = plen;
                if seqv[base as usize] == s {
                    if blen > 1 {
                        base = up[0][base as usize];
                    }
                    blen -= 1;
                }
                if blen > 0 && owner[base as usize] == x {
                    nn = base;
                    nl = blen;
                } else {
                    let id = owner.len() as u32;
                    owner.push(x);
                    seqv.push(s);
                    let p0 = if blen > 0 { base } else { id };
                    up[0].push(p0);
                    for j in 1..LOG {
                        let mid = up[j - 1][id as usize];
                        let a = up[j - 1][mid as usize];
                        up[j].push(a);
                    }
                    nn = id;
                    nl = blen + 1;
                }
            }
        }

        let newr = if rp > r { rp } else { r };

        if seqv[nn as usize] < newr {
            nl = 1;
        } else {
            let mut cnt: u32 = 1;
            let mut cur = nn;
            for j in (0..LOG).rev() {
                let step: u32 = 1u32 << j;
                if cnt + step <= nl {
                    let a = up[j][cur as usize];
                    if seqv[a as usize] >= newr {
                        cur = a;
                        cnt += step;
                    }
                }
            }
            nl = cnt;
        }

        vnode.push(nn);
        vlen.push(nl);
        vh.push(nh);
        vr.push(newr);

        out.push_str(&nl.to_string());
        out.push(' ');
        out.push_str(&nh.to_string());
        out.push(' ');
        out.push_str(&s.to_string());
        out.push(' ');
        out.push_str(&newr.to_string());
        out.push(' ');
        out.push_str(&owner[nn as usize].to_string());
        out.push(' ');

        if k > nl as u64 {
            out.push_str("NONE NONE");
        } else {
            let mut cur = nn;
            let d = k - 1;
            for j in 0..LOG {
                if (d >> j) & 1 == 1 {
                    cur = up[j][cur as usize];
                }
            }
            out.push_str(&owner[cur as usize].to_string());
            out.push(' ');
            out.push_str(&seqv[cur as usize].to_string());
        }
        out.push('\n');
    }

    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());
    w.write_all(out.as_bytes()).unwrap();
}