use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::{self, Read, Write};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let layout = it.next().unwrap().as_bytes()[0];
    let n: i64 = it.next().unwrap().parse().unwrap();
    let b: i64 = it.next().unwrap().parse().unwrap();
    let h: i64 = it.next().unwrap().parse().unwrap();
    let w: i64 = it.next().unwrap().parse().unwrap();
    let q: usize = it.next().unwrap().parse().unwrap();

    if n == 0 {
        println!("0");
        return;
    }

    let mut cache: HashMap<(i64, i64), u64> = HashMap::new();
    let mut heap: BinaryHeap<Reverse<(u64, i64, i64)>> = BinaryHeap::new();
    let mut tick: u64 = 0;

    let mut runs: Vec<(i64, i64, u64)> = Vec::new();

    let mut miss = |tr: i64, tc: i64,
                    cache: &mut HashMap<(i64, i64), u64>,
                    heap: &mut BinaryHeap<Reverse<(u64, i64, i64)>>,
                    tick: &mut u64,
                    runs: &mut Vec<(i64, i64, u64)>| {
        *tick += 1;
        let t = *tick;
        let key = (tr, tc);

        if let Some(v) = cache.get_mut(&key) {
            *v = t;
            heap.push(Reverse((t, tr, tc)));
            return;
        }

        if q == 0 {
            if let Some(last) = runs.last_mut() {
                if last.0 == tr && last.1 == tc {
                    last.2 += 1;
                    return;
                }
            }
            runs.push((tr, tc, 1));
            return;
        }

        if cache.len() == q {
            loop {
                if let Some(Reverse((ts, er, ec))) = heap.pop() {
                    if cache.get(&(er, ec)).copied() == Some(ts) {
                        cache.remove(&(er, ec));
                        break;
                    }
                }
            }
        }

        cache.insert(key, t);
        heap.push(Reverse((t, tr, tc)));

        if let Some(last) = runs.last_mut() {
            if last.0 == tr && last.1 == tc {
                last.2 += 1;
                return;
            }
        }
        runs.push((tr, tc, 1));
    };

    for c in 0..n {
        let dmax = std::cmp::min(b, n - 1 - c);

        if dmax >= 2 {
            if layout == b'L' {
                let mut d = dmax;
                while d >= 2 {
                    let tr = d / h;
                    let tc = c / w;
                    miss(tr, tc, &mut cache, &mut heap, &mut tick, &mut runs);

                    let low = std::cmp::max(2, tr * h);
                    d = low - 1;
                }
            } else {
                let mut d = dmax;
                while d >= 2 {
                    let tr = (b - d) / h;
                    let tc = (c + d) / w;
                    miss(tr, tc, &mut cache, &mut heap, &mut tick, &mut runs);

                    let row_low = b - (tr + 1) * h + 1;
                    let col_low = tc * w - c;
                    let low = std::cmp::max(2, std::cmp::max(row_low, col_low));
                    d = low - 1;
                }
            }
        }

        if layout == b'L' {
            let tr = 0;
            let tc = c / w;
            miss(tr, tc, &mut cache, &mut heap, &mut tick, &mut runs);

            if b >= 1 && c + 1 < n {
                miss(tr, tc, &mut cache, &mut heap, &mut tick, &mut runs);
            }
        } else {
            let tr = b / h;
            let tc = c / w;
            miss(tr, tc, &mut cache, &mut heap, &mut tick, &mut runs);

            if b >= 1 && c + 1 < n {
                let tr = (b - 1) / h;
                let tc = (c + 1) / w;
                miss(tr, tc, &mut cache, &mut heap, &mut tick, &mut runs);
            }
        }
    }

    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "{}", runs.len()).unwrap();
    for (r, c, cnt) in runs {
        writeln!(out, "{} {} {}", r, c, cnt).unwrap();
    }
}