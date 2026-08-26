use std::collections::HashMap;
use std::io::{self, Read, Write};

fn booth(a: &[i64]) -> usize {
    let n = a.len();
    if n <= 1 {
        return 0;
    }
    let mut i = 0usize;
    let mut j = 1usize;
    let mut k = 0usize;
    while i < n && j < n && k < n {
        let x = a[(i + k) % n];
        let y = a[(j + k) % n];
        if x == y {
            k += 1;
        } else if x > y {
            i += k + 1;
            if i <= j {
                i = j + 1;
            }
            k = 0;
        } else {
            j += k + 1;
            if j <= i {
                j = i + 1;
            }
            k = 0;
        }
    }
    i.min(j)
}

fn rotation_less(a: &[i64], sa: usize, b: &[i64], sb: usize) -> bool {
    let n = a.len();
    for k in 0..n {
        let x = a[(sa + k) % n];
        let y = b[(sb + k) % n];
        if x != y {
            return x < y;
        }
    }
    false
}

fn width(v: u64) -> u64 {
    if v <= 255 {
        1
    } else if v <= 65535 {
        2
    } else if v <= 16777215 {
        3
    } else {
        4
    }
}

fn round4(x: u64) -> u64 {
    (x + 3) / 4 * 4
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_ascii_whitespace();

    let n: usize = match it.next() {
        Some(s) => s.parse().unwrap(),
        None => return,
    };
    let r: i64 = it.next().unwrap().parse().unwrap();

    let mut a = Vec::with_capacity(n);
    for _ in 0..n {
        a.push(it.next().unwrap().parse::<i64>().unwrap());
    }

    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }

    a.resize(p, r);

    let w2 = width(r as u64);

    let mut best_size = u64::MAX;
    let mut best_b = 0usize;
    let mut best_u = 0usize;
    let mut best_w1 = 0u64;
    let mut best_refs: Vec<usize> = Vec::new();
    let mut best_reps: Vec<Vec<i64>> = Vec::new();

    let mut b = 1usize;
    while b <= p {
        let q = p / b;
        let mut map: HashMap<Vec<i64>, usize> = HashMap::with_capacity(q);
        let mut refs = Vec::with_capacity(q);
        let mut rev = vec![0i64; b];

        for block in 0..q {
            let start = block * b;
            let x = &a[start..start + b];

            for i in 0..b {
                rev[i] = x[b - 1 - i];
            }

            let s1 = booth(x);
            let s2 = booth(&rev);

            let d = (b - 1 - s2) % b;
            let t2 = b + d;

            let use_reverse = rotation_less(&rev, s2, x, s1);

            let mut key = Vec::with_capacity(b);
            let t;
            if use_reverse {
                t = t2;
                for i in 0..b {
                    key.push(rev[(s2 + i) % b]);
                }
            } else {
                t = s1;
                for i in 0..b {
                    key.push(x[(s1 + i) % b]);
                }
            }

            let id = if let Some(&id) = map.get(&key) {
                id
            } else {
                let id = map.len();
                map.insert(key, id);
                id
            };

            refs.push(2 * b * id + t);
        }

        let u = map.len();
        let w1 = width((2u64 * b as u64 * u as u64).saturating_sub(1));
        let size = round4(q as u64 * w1) + round4(u as u64 * b as u64 * w2);

        if size < best_size || (size == best_size && b < best_b) {
            let mut reps: Vec<Vec<i64>> = (0..u).map(|_| Vec::new()).collect();
            for (key, id) in map {
                reps[id] = key;
            }

            best_size = size;
            best_b = b;
            best_u = u;
            best_w1 = w1;
            best_refs = refs;
            best_reps = reps;
        }

        b <<= 1;
    }

    let mut out = io::BufWriter::new(io::stdout().lock());

    writeln!(
        out,
        "{} {} {} {} {}",
        best_size, best_b, best_u, best_w1, w2
    )
    .unwrap();

    write!(out, "{}", best_refs.len()).unwrap();
    for v in best_refs {
        write!(out, " {}", v).unwrap();
    }
    writeln!(out).unwrap();

    for rep in best_reps {
        for (i, v) in rep.iter().enumerate() {
            if i > 0 {
                write!(out, " ").unwrap();
            }
            write!(out, "{}", v).unwrap();
        }
        writeln!(out).unwrap();
    }
}