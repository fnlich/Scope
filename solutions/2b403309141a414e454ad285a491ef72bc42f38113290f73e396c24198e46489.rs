use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};

fn insert_interval(map: &mut BTreeMap<i64, i64>, mut l: i64, mut r: i64) {
    if let Some((&pl, &pr)) = map.range(..=l).next_back() {
        if pr + 1 >= l {
            l = pl.min(l);
            r = r.max(pr);
            map.remove(&pl);
        }
    }

    let keys: Vec<i64> = map
        .range(l..)
        .take_while(|(&sl, &sr)| sl <= r + 1)
        .map(|(&sl, &sr)| {
            r = r.max(sr);
            sl
        })
        .collect();

    for k in keys {
        map.remove(&k);
    }

    map.insert(l, r);
}

fn remove_interval(map: &mut BTreeMap<i64, i64>, l: i64, r: i64) {
    let mut affected = Vec::new();

    if let Some((&sl, &sr)) = map.range(..=l).next_back() {
        if sr >= l {
            affected.push((sl, sr));
        }
    }

    for (&sl, &sr) in map.range(l..=r) {
        if affected.last().map_or(true, |&(x, _)| x != sl) {
            affected.push((sl, sr));
        }
    }

    for &(sl, _) in &affected {
        map.remove(&sl);
    }

    for (sl, sr) in affected {
        if sl < l {
            map.insert(sl, l - 1);
        }
        if sr > r {
            map.insert(r + 1, sr);
        }
    }
}

fn first_occupied(map: &BTreeMap<i64, i64>, l: i64, r: i64) -> Option<i64> {
    if let Some((&sl, &sr)) = map.range(..=l).next_back() {
        if sr >= l {
            return Some(l);
        }
    }

    if let Some((&sl, _)) = map.range(l..=r).next() {
        if sl <= r {
            return Some(sl);
        }
    }

    None
}

fn first_fault(map: &BTreeMap<i64, (i64, i64)>, l: i64, r: i64) -> Option<(i64, i64)> {
    if let Some((&fl, &(fr, code))) = map.range(..=l).next_back() {
        if fr >= l {
            return Some((l, code));
        }
    }

    if let Some((&fl, &(fr, code))) = map.range(l..=r).next() {
        if fl <= r {
            return Some((fl, code));
        }
    }

    None
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_ascii_whitespace();

    let p: usize = it.next().unwrap().parse().unwrap();
    let f: usize = it.next().unwrap().parse().unwrap();
    let q: usize = it.next().unwrap().parse().unwrap();

    let mut occupied_intervals = Vec::with_capacity(p);
    for _ in 0..p {
        let l: i64 = it.next().unwrap().parse().unwrap();
        let r: i64 = it.next().unwrap().parse().unwrap();
        occupied_intervals.push((l, r));
    }

    occupied_intervals.sort_unstable();

    let mut occupied = BTreeMap::<i64, i64>::new();
    for (l, r) in occupied_intervals {
        insert_interval(&mut occupied, l, r);
    }

    let mut faults = BTreeMap::<i64, (i64, i64)>::new();
    for _ in 0..f {
        let l: i64 = it.next().unwrap().parse().unwrap();
        let r: i64 = it.next().unwrap().parse().unwrap();
        let c: i64 = it.next().unwrap().parse().unwrap();
        faults.insert(l, (r, c));
    }

    let mut allocations = HashMap::<i64, (i64, i64)>::new();
    let mut output = io::BufWriter::new(io::stdout());

    for _ in 0..q {
        let op = it.next().unwrap();

        if op == "R" {
            let id: i64 = it.next().unwrap().parse().unwrap();
            if let Some((l, r)) = allocations.remove(&id) {
                remove_interval(&mut occupied, l, r);
            }
            continue;
        }

        let id: i64 = it.next().unwrap().parse().unwrap();
        let b: i64 = it.next().unwrap().parse().unwrap();
        let k: i64 = it.next().unwrap().parse().unwrap();
        let mut budget: i64 = it.next().unwrap().parse().unwrap();

        let max_start = 65536 - k;
        let mut s = b;
        let mut failed = false;

        loop {
            if s > max_start {
                writeln!(output, "FAIL {} RANGE {}", id, s).unwrap();
                failed = true;
                break;
            }

            let end = s + k - 1;

            if let Some(p_occ) = first_occupied(&occupied, s, end) {
                let bad_count = p_occ - s + 1;

                if budget <= bad_count {
                    let next = s + budget;
                    if next > max_start {
                        writeln!(output, "FAIL {} RANGE {}", id, next).unwrap();
                    } else {
                        writeln!(output, "FAIL {} BUDGET {}", id, next).unwrap();
                    }
                    failed = true;
                    break;
                }

                budget -= bad_count;
                s = p_occ + 1;
                continue;
            }

            if let Some((fault_port, code)) = first_fault(&faults, s, end) {
                writeln!(output, "FAIL {} BIND {} {}", id, fault_port, code).unwrap();
                failed = true;
                break;
            }

            insert_interval(&mut occupied, s, end);
            allocations.insert(id, (s, end));
            writeln!(output, "OK {} {}", id, s).unwrap();
            break;
        }

        if failed {
            break;
        }
    }

    if !failed_any(&output) {
    }
}