use std::io::{self, Read, Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_ascii_whitespace();
    let n: usize = it.next().unwrap().parse().unwrap();
    let m: usize = it.next().unwrap().parse().unwrap();
    let t: usize = it.next().unwrap().parse().unwrap();
    let s0 = it.next().unwrap();
    let init_legacy = s0 == "LEGACY";

    let mut kind = vec![0u8; n + 1];
    let mut dur = vec![0i64; n + 1];
    let mut dur2 = vec![0i64; n + 1];
    let mut out1 = vec![0u8; n + 1];
    let mut out2 = vec![0u8; n + 1];
    let mut rback = vec![false; n + 1];

    let ocode = |x: &str| -> u8 {
        match x {
            "OK" => 0,
            "DUP" => 1,
            "MISS" => 2,
            "ERR" => 3,
            "EMPTY" => 4,
            "BADCUSTOMER" => 5,
            "UNSUPPORTED" => 6,
            _ => 3,
        }
    };

    for i in 1..=n {
        let k = it.next().unwrap();
        match k {
            "C" => {
                kind[i] = 0;
                dur[i] = it.next().unwrap().parse().unwrap();
                out1[i] = ocode(it.next().unwrap());
            }
            "R" => {
                kind[i] = 1;
                dur[i] = it.next().unwrap().parse().unwrap();
                rback[i] = it.next().unwrap() == "LEGACY";
                out1[i] = ocode(it.next().unwrap());
            }
            "D" => {
                kind[i] = 2;
                dur[i] = it.next().unwrap().parse().unwrap();
                let mode = it.next().unwrap();
                out2[i] = if mode == "REQUIRE" { 1 } else { 0 };
                out1[i] = ocode(it.next().unwrap());
            }
            _ => {
                kind[i] = 3;
                dur[i] = it.next().unwrap().parse().unwrap();
                out1[i] = ocode(it.next().unwrap());
                dur2[i] = it.next().unwrap().parse().unwrap();
                out2[i] = ocode(it.next().unwrap());
            }
        }
    }

    let mut eu = vec![0u32; m];
    let mut ev = vec![0u32; m];
    let mut headc = vec![0u32; n + 2];
    let mut revc = vec![0u32; n + 2];
    for e in 0..m {
        let u: usize = it.next().unwrap().parse().unwrap();
        let v: usize = it.next().unwrap().parse().unwrap();
        eu[e] = u as u32;
        ev[e] = v as u32;
        headc[u] += 1;
        revc[v] += 1;
    }
    let mut hs = vec![0u32; n + 2];
    let mut rs = vec![0u32; n + 2];
    for i in 1..=n {
        hs[i + 1] = hs[i] + headc[i];
        rs[i + 1] = rs[i] + revc[i];
    }
    let mut fwd = vec![0u32; m];
    let mut rev = vec![0u32; m];
    let mut hp = hs.clone();
    let mut rp = rs.clone();
    for e in 0..m {
        let u = eu[e] as usize;
        let v = ev[e] as usize;
        fwd[hp[u] as usize] = v as u32;
        hp[u] += 1;
        rev[rp[v] as usize] = u as u32;
        rp[v] += 1;
    }

    let mut req = vec![false; n + 1];
    let mut stack = vec![t];
    req[t] = true;
    while let Some(x) = stack.pop() {
        for j in rs[x] as usize..rs[x + 1] as usize {
            let p = rev[j] as usize;
            if !req[p] {
                req[p] = true;
                stack.push(p);
            }
        }
    }

    let mut indeg = vec![0u32; n + 1];
    for i in 1..=n {
        if req[i] {
            let mut c = 0;
            for j in rs[i] as usize..rs[i + 1] as usize {
                if req[rev[j] as usize] {
                    c += 1;
                }
            }
            indeg[i] = c;
        }
    }

    let mut dwait: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    let mut vwait: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    let mut events: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    let mut legacy = init_legacy;
    let mut dbusy = false;
    let mut vbusy = false;
    let mut startlegacy = vec![false; n + 1];

    let mut ready_par: Vec<usize> = Vec::new();
    for i in 1..=n {
        if req[i] && indeg[i] == 0 {
            match kind[i] {
                0 | 1 => ready_par.push(i),
                2 => dwait.push(Reverse(i)),
                _ => vwait.push(Reverse(i)),
            }
        }
    }
    for i in ready_par {
        events.push(Reverse((dur[i], i)));
    }
    if !dbusy {
        if let Some(Reverse(i)) = dwait.pop() {
            dbusy = true;
            events.push(Reverse((dur[i], i)));
        }
    }
    if !vbusy {
        if let Some(Reverse(i)) = vwait.pop() {
            vbusy = true;
            startlegacy[i] = legacy;
            let d = if legacy { dur2[i] } else { dur[i] };
            events.push(Reverse((d, i)));
        }
    }

    let stdout = io::stdout();
    let mut o = io::BufWriter::new(stdout.lock());

    let mut done_time: Option<i64> = None;

    while let Some(&Reverse((tm, _))) = events.peek() {
        let mut batch: Vec<usize> = Vec::new();
        while let Some(&Reverse((tt, i))) = events.peek() {
            if tt == tm {
                batch.push(i);
                events.pop();
            } else {
                break;
            }
        }
        batch.sort_unstable();
        for &i in &batch {
            if kind[i] == 2 {
                dbusy = false;
            } else if kind[i] == 3 {
                vbusy = false;
            }
        }
        let mut fail: Option<(usize, &str)> = None;
        for &i in &batch {
            let r: Option<&str> = match kind[i] {
                0 | 1 => {
                    if out1[i] == 3 {
                        Some("INTERNAL")
                    } else {
                        None
                    }
                }
                2 => {
                    if out1[i] == 3 {
                        Some("INTERNAL")
                    } else if out1[i] == 2 && out2[i] == 1 {
                        Some("NOT_FOUND")
                    } else {
                        None
                    }
                }
                _ => {
                    let c = if startlegacy[i] { out2[i] } else { out1[i] };
                    match c {
                        0 | 1 => None,
                        3 => Some("INTERNAL"),
                        4 => Some("EMPTY_PAYLOAD"),
                        5 => Some("BAD_CUSTOMER"),
                        6 => Some("UNSUPPORTED"),
                        _ => Some("INTERNAL"),
                    }
                }
            };
            if let Some(rr) = r {
                if fail.is_none() {
                    fail = Some((i, rr));
                }
            }
        }
        if let Some((i, rr)) = fail {
            writeln!(o, "FAIL {} {} {}", tm, i, rr).unwrap();
            return;
        }
        for &i in &batch {
            if kind[i] == 1 {
                legacy = rback[i];
            }
            if i == t {
                done_time = Some(tm);
            }
        }
        if done_time.is_some() {
            writeln!(o, "DONE {}", done_time.unwrap()).unwrap();
            return;
        }
        let mut newpar: Vec<usize> = Vec::new();
        for &i in &batch {
            for j in hs[i] as usize..hs[i + 1] as usize {
                let v = fwd[j] as usize;
                if req[v] {
                    indeg[v] -= 1;
                    if indeg[v] == 0 {
                        match kind[v] {
                            0 | 1 => newpar.push(v),
                            2 => dwait.push(Reverse(v)),
                            _ => vwait.push(Reverse(v)),
                        }
                    }
                }
            }
        }
        for i in newpar {
            events.push(Reverse((tm + dur[i], i)));
        }
        if !dbusy {
            if let Some(Reverse(i)) = dwait.pop() {
                dbusy = true;
                events.push(Reverse((tm + dur[i], i)));
            }
        }
        if !vbusy {
            if let Some(Reverse(i)) = vwait.pop() {
                vbusy = true;
                startlegacy[i] = legacy;
                let d = if legacy { dur2[i] } else { dur[i] };
                events.push(Reverse((tm + d, i)));
            }
        }
    }
}