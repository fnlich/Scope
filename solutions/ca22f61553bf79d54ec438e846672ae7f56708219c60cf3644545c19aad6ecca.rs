use std::io::{self, Read, Write};
use std::collections::BTreeSet;

fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let toks: Vec<&str> = s.split_ascii_whitespace().collect();
    let mut ti = 0usize;
    let mut next = || -> &str { let t = toks[ti]; ti += 1; t };
    let _ = &next;
    let mut idx = 0usize;
    let mut tk = |i: &mut usize| -> &str { let t = toks[*i]; *i += 1; t };
    let n = usize::from_str_radix(tk(&mut idx), 16).unwrap();
    let q = usize::from_str_radix(tk(&mut idx), 16).unwrap();
    let d = u32::from_str_radix(tk(&mut idx), 16).unwrap();

    let mut cands: Vec<Vec<u32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let k = usize::from_str_radix(tk(&mut idx), 16).unwrap();
        let mut v = Vec::with_capacity(k);
        for _ in 0..k {
            v.push(u32::from_str_radix(tk(&mut idx), 16).unwrap());
        }
        cands.push(v);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| cands[a].cmp(&cands[b]));
    let mut rank = vec![0usize; n];
    for (p, &c) in order.iter().enumerate() {
        rank[c] = p;
    }
    let mut enabled: BTreeSet<usize> = (0..n).collect();

    let m = usize::from_str_radix(tk(&mut idx), 16).unwrap();
    let p0 = usize::from_str_radix(tk(&mut idx), 16).unwrap();
    let mut text: Vec<u32> = Vec::with_capacity(m);
    for _ in 0..m {
        text.push(u32::from_str_radix(tk(&mut idx), 16).unwrap());
    }
    let mut left: Vec<u32> = text[..p0].to_vec();
    let mut right: Vec<u32> = Vec::with_capacity(m - p0);
    for i in (p0..m).rev() {
        right.push(text[i]);
    }
    let mut ldel: Vec<usize> = Vec::new();
    for (i, &c) in left.iter().enumerate() {
        if c == d {
            ldel.push(i);
        }
    }
    let mut rdel: Vec<usize> = Vec::new();
    for (i, &c) in right.iter().enumerate() {
        if c == d {
            rdel.push(i);
        }
    }

    let mut active = false;
    let mut lo = 0usize;
    let mut hi = 0usize;
    let mut pos = 0usize;
    let mut committed: usize = 0;

    let mut out = String::new();

    let mut qq = q;
    let mut finished = false;

    while qq > 0 && !finished {
        qq -= 1;
        let cmd = tk(&mut idx);
        match cmd {
            "A" => {
                let id = usize::from_str_radix(tk(&mut idx), 16).unwrap() - 1;
                let r = rank[id];
                if enabled.contains(&r) {
                    enabled.remove(&r);
                    if active && pos == r {
                        let nx = enabled
                            .range(pos..hi)
                            .next()
                            .copied()
                            .or_else(|| enabled.range(lo..hi).next().copied());
                        match nx {
                            Some(v) => pos = v,
                            None => active = false,
                        }
                    }
                } else {
                    enabled.insert(r);
                }
            }
            "C" => {
                active = false;
            }
            "T" | "P" => {
                let fwd = cmd == "T";
                if active {
                    if fwd {
                        let nx = enabled
                            .range(pos + 1..hi)
                            .next()
                            .copied()
                            .or_else(|| enabled.range(lo..hi).next().copied());
                        if let Some(v) = nx {
                            pos = v;
                        }
                    } else {
                        let nx = enabled
                            .range(lo..pos)
                            .next_back()
                            .copied()
                            .or_else(|| enabled.range(lo..hi).next_back().copied());
                        if let Some(v) = nx {
                            pos = v;
                        }
                    }
                } else {
                    let a = match ldel.last() {
                        Some(&j) => j + 1,
                        None => 0,
                    };
                    let cur = left.len();
                    let mut pre: &[u32] = &left[a..cur];
                    let mut st = 0usize;
                    let mut en = pre.len();
                    while st < en && pre[st] == 0x20 {
                        st += 1;
                    }
                    while en > st && pre[en - 1] == 0x20 {
                        en -= 1;
                    }
                    pre = &pre[st..en];
                    let l = {
                        let mut a2 = 0usize;
                        let mut b2 = n;
                        while a2 < b2 {
                            let mid = (a2 + b2) / 2;
                            if cands[order[mid]].as_slice() < pre {
                                a2 = mid + 1;
                            } else {
                                b2 = mid;
                            }
                        }
                        a2
                    };
                    let h = {
                        let mut a2 = 0usize;
                        let mut b2 = n;
                        while a2 < b2 {
                            let mid = (a2 + b2) / 2;
                            let c = &cands[order[mid]];
                            let ok = c.as_slice() < pre
                                || (c.len() >= pre.len() && &c[..pre.len()] == pre);
                            if ok {
                                a2 = mid + 1;
                            } else {
                                b2 = mid;
                            }
                        }
                        a2
                    };
                    if l < h {
                        let pick = if fwd {
                            enabled.range(l..h).next().copied()
                        } else {
                            enabled.range(l..h).next_back().copied()
                        };
                        if let Some(v) = pick {
                            active = true;
                            lo = l;
                            hi = h;
                            pos = v;
                        }
                    }
                }
            }
            _ => {
                let mut did_commit = false;
                if active {
                    let a = match ldel.last() {
                        Some(&j) => j + 1,
                        None => 0,
                    };
                    let cur = left.len();
                    let total = left.len() + right.len();
                    let b = match rdel.last() {
                        Some(&r) => total - 1 - r,
                        None => total,
                    };
                    let cnt = b - cur;
                    while left.len() > a {
                        let c = left.pop().unwrap();
                        if c == d {
                            ldel.pop();
                        }
                    }
                    for _ in 0..cnt {
                        let c = right.pop().unwrap();
                        if c == d {
                            rdel.pop();
                        }
                    }
                    let cid = order[pos];
                    for &c in cands[cid].iter() {
                        left.push(c);
                    }
                    committed = cid + 1;
                    active = false;
                    did_commit = true;
                }
                let _ = did_commit;
                match cmd {
                    "I" => {
                        let x = u32::from_str_radix(tk(&mut idx), 16).unwrap();
                        left.push(x);
                        if x == d {
                            ldel.push(left.len() - 1);
                        }
                        committed = 0;
                    }
                    "B" => {
                        if !left.is_empty() {
                            let c = left.pop().unwrap();
                            if c == d {
                                ldel.pop();
                            }
                        }
                        committed = 0;
                    }
                    "L" => {
                        if !left.is_empty() {
                            let c = left.pop().unwrap();
                            if c == d {
                                ldel.pop();
                            }
                            right.push(c);
                            if c == d {
                                rdel.push(right.len() - 1);
                            }
                        }
                        committed = 0;
                    }
                    "R" => {
                        if !right.is_empty() {
                            let c = right.pop().unwrap();
                            if c == d {
                                rdel.pop();
                            }
                            left.push(c);
                            if c == d {
                                ldel.push(left.len() - 1);
                            }
                        }
                        committed = 0;
                    }
                    "E" => {
                        finished = true;
                    }
                    _ => {}
                }
            }
        }
    }

    let total = left.len() + right.len();
    out.push_str(&format!("{:X}", total));
    for &c in left.iter() {
        out.push(' ');
        out.push_str(&format!("{:X}", c));
    }
    for i in (0..right.len()).rev() {
        out.push(' ');
        out.push_str(&format!("{:X}", right[i]));
    }
    out.push('\n');
    out.push_str(&format!("{}\n", committed));

    let so = io::stdout();
    let mut w = io::BufWriter::new(so.lock());
    w.write_all(out.as_bytes()).unwrap();
}