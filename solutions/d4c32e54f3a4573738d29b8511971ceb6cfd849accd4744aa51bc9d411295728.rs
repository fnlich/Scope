use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_ascii_whitespace();

    let m: u64 = it.next().unwrap().parse().unwrap();
    let mut s: u64 = it.next().unwrap().parse().unwrap();
    let b: u64 = it.next().unwrap().parse().unwrap();
    let k: usize = it.next().unwrap().parse().unwrap();

    struct Seg {
        q: u64,
        l: u64,
        dl: u64,
        u: Option<u64>,
        du: u64,
        a: u64,
        da: u64,
    }

    let mut segs = Vec::with_capacity(k);
    for _ in 0..k {
        let q: u64 = it.next().unwrap().parse().unwrap();
        let l: u64 = it.next().unwrap().parse().unwrap();
        let dl: u64 = it.next().unwrap().parse().unwrap();
        let us = it.next().unwrap();
        let u = if us == "-1" {
            None
        } else {
            Some(us.parse::<u64>().unwrap())
        };
        let du: u64 = it.next().unwrap().parse().unwrap();
        let a: u64 = it.next().unwrap().parse().unwrap();
        let da: u64 = it.next().unwrap().parse().unwrap();
        segs.push(Seg { q, l, dl, u, du, a, da });
    }

    let mut capacity = s;
    let mut length = 0u64;
    let mut payload = 0u64;
    let mut records: Vec<(u64, u64, u64)> = Vec::new();

    let mut terminal = String::new();
    let mut stopped = false;

    'outer: for seg in segs {
        let mut j = 0u64;

        while j < seg.q {
            if length == capacity {
                let lo128 = seg.l as u128 + seg.dl as u128 * j as u128;
                let lo = lo128 as u64;

                let base128 = capacity as u128 + 1 + lo as u128;
                if base128 > u64::MAX as u128 {
                    terminal = format!("OVERFLOW {}", length);
                    stopped = true;
                    break 'outer;
                }
                let base = base128 as u64;

                let g = capacity.saturating_mul(2);

                let raw = if let Some(u) = seg.u {
                    let hi128 = u as u128 + seg.du as u128 * j as u128;
                    let hi = hi128 as u64;
                    let ceiling128 = capacity as u128 + 1 + hi as u128;
                    if ceiling128 > u64::MAX as u128 {
                        terminal = format!("OVERFLOW {}", length);
                        stopped = true;
                        break 'outer;
                    }
                    let ceiling = ceiling128 as u64;
                    base.max(g.min(ceiling))
                } else {
                    base.max(g)
                };

                let rem = raw % b;
                let add = if rem == 0 { 0 } else { b - rem };
                let actual128 = raw as u128 + add as u128;
                if actual128 > u64::MAX as u128 {
                    terminal = format!("OVERFLOW {}", length);
                    stopped = true;
                    break 'outer;
                }
                let actual = actual128 as u64;

                let live = payload as u128 + capacity as u128 + actual as u128;
                if live > m as u128 {
                    terminal = format!("ALLOC {} {} {}", length, raw, actual);
                    stopped = true;
                    break 'outer;
                }

                records.push((length, raw, actual));
                capacity = actual;
            }

            let remaining = seg.q - j;
            let until_reservation = capacity - length;
            let batch = remaining.min(until_reservation);

            if batch > 0 {
                let base_x = seg.a as u128 + seg.da as u128 * j as u128;
                let step = seg.da as u128;

                let can_clone = |cnt: u64| -> bool {
                    if cnt == 0 {
                        return true;
                    }
                    let n = cnt as u128;
                    let total = n * base_x + step * n * (n - 1) / 2;
                    payload as u128 + total + capacity as u128 <= m as u128
                };

                if !can_clone(batch) {
                    let mut lo = 0u64;
                    let mut hi = batch;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if can_clone(mid + 1) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }

                    if lo > 0 {
                        let n = lo as u128;
                        let total = n * base_x + step * n * (n - 1) / 2;
                        payload = (payload as u128 + total) as u64;
                        length += lo;
                        j += lo;
                    }

                    let x = (seg.a as u128 + seg.da as u128 * j as u128) as u64;
                    terminal = format!("CLONE {} {}", length, x);
                    stopped = true;
                    break 'outer;
                } else {
                    let n = batch as u128;
                    let total = n * base_x + step * n * (n - 1) / 2;
                    payload = (payload as u128 + total) as u64;
                    length += batch;
                    j += batch;
                }
            }
        }
    }

    if !stopped {
        terminal = format!("OK {} {} {}", length, capacity, payload);
    }

    let mut out = String::new();
    out.push_str(&format!("{}\n", records.len()));
    for (p, raw, actual) in records {
        out.push_str(&format!("{} {} {}\n", p, raw, actual));
    }
    out.push_str(&terminal);
    println!("{}", out);
}