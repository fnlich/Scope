use std::collections::BTreeMap;
use std::io::{self, Read, Write};

#[derive(Clone, Copy)]
struct Style {
    as_: i128,
    ns: i128,
    ao: i128,
    no: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.abs()
}

fn fraction(a: i128, b: i128, d: i128, dist: i128) -> String {
    let num = a * d + (b - a) * dist;
    let g = gcd(num, d);
    format!("{}/{}", num / g, d / g)
}

fn split_at(map: &mut BTreeMap<i128, Style>, x: i128, n: i128) {
    if x <= 0 || x >= n || map.contains_key(&x) {
        return;
    }
    if let Some((&k, &v)) = map.range(..=x).next_back() {
        if k < x {
            map.insert(x, v);
        }
    }
}

fn assign(map: &mut BTreeMap<i128, Style>, l: i128, r: i128, n: i128, style: Style) {
    if l <= r {
        split_at(map, l, n);
        if r + 1 < n {
            split_at(map, r + 1, n);
        }
        let keys: Vec<i128> = map.range(l..=r).map(|(&k, _)| k).collect();
        for k in keys {
            map.remove(&k);
        }
        map.insert(l, style);
    } else {
        assign(map, l, n - 1, n, style);
        assign(map, 0, r, n, style);
    }
}

fn get_style(map: &BTreeMap<i128, Style>, x: i128) -> Style {
    *map.range(..=x).next_back().unwrap().1
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let n: i128 = it.next().unwrap().parse::<u64>().unwrap() as i128;
    let d: i128 = it.next().unwrap().parse::<u64>().unwrap() as i128;
    let q: usize = it.next().unwrap().parse().unwrap();

    let initial = Style {
        as_: it.next().unwrap().parse::<u64>().unwrap() as i128,
        ns: it.next().unwrap().parse::<u64>().unwrap() as i128,
        ao: it.next().unwrap().parse::<u64>().unwrap() as i128,
        no: it.next().unwrap().parse::<u64>().unwrap() as i128,
    };

    let total = n * d;
    let mut pos: i128 = 0;
    let mut dir: i128 = 1;

    let mut styles = BTreeMap::<i128, Style>::new();
    styles.insert(0, initial);

    let mut out = io::BufWriter::new(io::stdout().lock());

    for _ in 0..q {
        match it.next().unwrap() {
            "SWIPE" => {
                let x: i128 = it.next().unwrap().parse::<i64>().unwrap() as i128;
                pos = (pos + x).rem_euclid(total);
                if x > 0 {
                    dir = 1;
                } else if x < 0 {
                    dir = -1;
                }
            }
            "SNAP" => {
                if n > 1 {
                    let k = pos.div_euclid(d);
                    let rem = pos.rem_euclid(d);
                    let mut target = k;
                    if rem * 2 > d || (rem * 2 == d && dir > 0) {
                        target += 1;
                    }
                    target = target.rem_euclid(n);
                    pos = target * d;
                } else {
                    pos = 0;
                }
            }
            "STYLE" => {
                let l: i128 = it.next().unwrap().parse::<u64>().unwrap() as i128;
                let r: i128 = it.next().unwrap().parse::<u64>().unwrap() as i128;
                let style = Style {
                    as_: it.next().unwrap().parse::<u64>().unwrap() as i128,
                    ns: it.next().unwrap().parse::<u64>().unwrap() as i128,
                    ao: it.next().unwrap().parse::<u64>().unwrap() as i128,
                    no: it.next().unwrap().parse::<u64>().unwrap() as i128,
                };
                assign(&mut styles, l, r, n, style);
            }
            "ASK" => {
                let j: i128 = it.next().unwrap().parse::<u64>().unwrap() as i128;

                if n == 1 {
                    let s = get_style(&styles, 0);
                    writeln!(
                        out,
                        "C {} {}",
                        fraction(s.as_, s.as_, d, 0),
                        fraction(s.ao, s.ao, d, 0)
                    ).unwrap();
                    continue;
                }

                let center = j * d;
                let raw = center - pos;
                let rem = raw.rem_euclid(total);

                let signed = if rem * 2 < total {
                    rem
                } else if rem * 2 > total {
                    rem - total
                } else if dir > 0 {
                    rem
                } else {
                    rem - total
                };

                let dist = signed.abs();

                if dist > d {
                    writeln!(out, "H").unwrap();
                    continue;
                }

                let s = get_style(&styles, j);
                let side = if signed == 0 {
                    "C"
                } else if signed < 0 {
                    "L"
                } else {
                    "R"
                };

                let scale = fraction(s.as_, s.ns, d, dist);
                let opacity = fraction(s.ao, s.no, d, dist);

                writeln!(out, "{} {} {}", side, scale, opacity).unwrap();
            }
            _ => unreachable!(),
        }
    }
}