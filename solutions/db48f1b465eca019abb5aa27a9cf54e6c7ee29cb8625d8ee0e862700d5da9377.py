use std::io::{self, Read, Write};
use std::collections::BTreeMap;

fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_ascii_whitespace();
    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    let n: usize = match it.next() {
        Some(v) => v.parse().unwrap(),
        None => {
            writeln!(w, "0").unwrap();
            return;
        }
    };

    struct Snap {
        m: i64,
        removes: Vec<u64>,
        adds: Vec<(u64, u64, u64)>,
        seq: u64,
    }

    let mut snaps: Vec<Snap> = Vec::with_capacity(n);

    for _ in 0..n {
        let q: usize = it.next().unwrap().parse().unwrap();
        let seq: u64 = it.next().unwrap().parse().unwrap();
        let sm: i64 = it.next().unwrap().parse().unwrap();
        let pm: i64 = it.next().unwrap().parse().unwrap();
        let eff: i64 = if pm != -1 { pm } else { sm };
        let mut removes: Vec<u64> = Vec::new();
        let mut adds: Vec<(u64, u64, u64)> = Vec::new();
        for _ in 0..q {
            let k = it.next().unwrap();
            if k == "R" {
                let id: u64 = it.next().unwrap().parse().unwrap();
                removes.push(id);
            } else {
                let id: u64 = it.next().unwrap().parse().unwrap();
                let size: u64 = it.next().unwrap().parse().unwrap();
                let d: u64 = it.next().unwrap().parse().unwrap();
                adds.push((id, size, d));
            }
        }
        if eff == -1 {
            continue;
        }
        snaps.push(Snap {
            m: eff,
            removes,
            adds,
            seq,
        });
    }

    if snaps.is_empty() {
        writeln!(w, "0").unwrap();
        return;
    }

    let first = snaps[0].m;
    let last = snaps[snaps.len() - 1].m;

    const P: u128 = 1000000007;
    const C: u128 = 1000003;
    const B: u128 = 911382323;

    let mut live: BTreeMap<u64, (u64, u64, i64)> = BTreeMap::new();
    let mut expiries: BTreeMap<i64, Vec<u64>> = BTreeMap::new();

    let mut idx = 0usize;
    let mut lines: Vec<String> = Vec::new();
    let mut count = 0usize;

    let mut month = first;
    while month <= last {
        if let Some(list) = expiries.remove(&month) {
            for id in list {
                let mut drop = false;
                if let Some(e) = live.get(&id) {
                    if e.2 == month {
                        drop = true;
                    }
                }
                if drop {
                    live.remove(&id);
                }
            }
        }
        while idx < snaps.len() && snaps[idx].m == month {
            let sn = &snaps[idx];
            for id in sn.removes.iter() {
                live.remove(id);
            }
            for (id, size, d) in sn.adds.iter() {
                let exp = month + (*d as i64);
                live.insert(*id, (*size, sn.seq, exp));
                expiries.entry(exp).or_insert_with(Vec::new).push(*id);
            }
            idx += 1;
        }

        let mut h: u128 = 0;
        let mut total: u128 = 0;
        let mut cnt: u64 = 0;
        for (id, (size, seq, _)) in live.iter() {
            let a = ((*id as u128) % P) * C % P;
            let a = (a + (*size as u128) % P) % P;
            let a = a * C % P;
            let v = (a + (*seq as u128) % P) % P;
            h = (h * B + v) % P;
            total += *size as u128;
            cnt += 1;
        }
        lines.push(format!("{} {} {} {}", month, cnt, total, h));
        count += 1;
        month += 1;
    }

    writeln!(w, "{}", count).unwrap();
    for l in lines {
        writeln!(w, "{}", l).unwrap();
    }
}