use std::collections::HashMap;
use std::io::{self, Read, Write};

#[derive(Clone, Copy)]
struct Op {
    kind: u8,
    a: i64,
    b: i64,
}

struct Fenwick {
    bit: Vec<[i32; 5]>,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            bit: vec![[0; 5]; n + 1],
        }
    }

    fn add(&mut self, mut i: usize, reason: usize, delta: i32) {
        while i < self.bit.len() {
            self.bit[i][reason] += delta;
            i += i & (!i + 1);
        }
    }

    fn sum(&self, mut i: usize) -> [i64; 5] {
        let mut res = [0i64; 5];
        while i > 0 {
            for j in 0..5 {
                res[j] += self.bit[i][j] as i64;
            }
            i &= i - 1;
        }
        res
    }

    fn range(&self, left_count: usize, right_count: usize) -> [i64; 5] {
        let a = self.sum(right_count);
        let b = self.sum(left_count);
        [
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
            a[3] - b[3],
            a[4] - b[4],
        ]
    }
}

fn lower_bound(a: &[i64], x: i64) -> usize {
    let mut l = 0;
    let mut r = a.len();
    while l < r {
        let m = (l + r) / 2;
        if a[m] < x {
            l = m + 1;
        } else {
            r = m;
        }
    }
    l
}

fn upper_bound(a: &[i64], x: i64) -> usize {
    let mut l = 0;
    let mut r = a.len();
    while l < r {
        let m = (l + r) / 2;
        if a[m] <= x {
            l = m + 1;
        } else {
            r = m;
        }
    }
    l
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let q: usize = it.next().unwrap().parse().unwrap();
    let mut ops = Vec::with_capacity(q);
    let mut coords = Vec::with_capacity(q * 2);

    for _ in 0..q {
        let s = it.next().unwrap();
        let (kind, a, b) = match s.as_bytes()[0] {
            b'N' => {
                let h = it.next().unwrap().parse::<i64>().unwrap();
                (0, h, 0)
            }
            b'M' => {
                let h = it.next().unwrap().parse::<i64>().unwrap();
                let k = it.next().unwrap().parse::<i64>().unwrap();
                (1, h, k)
            }
            b'D' => {
                let h = it.next().unwrap().parse::<i64>().unwrap();
                (2, h, 0)
            }
            b'C' => {
                if s == "CANCEL" {
                    let h = it.next().unwrap().parse::<i64>().unwrap();
                    (3, h, 0)
                } else {
                    (6, 0, 0)
                }
            }
            b'R' => {
                let h = it.next().unwrap().parse::<i64>().unwrap();
                (4, h, 0)
            }
            b'W' => {
                let l = it.next().unwrap().parse::<i64>().unwrap();
                let r = it.next().unwrap().parse::<i64>().unwrap();
                (5, l, r)
            }
            _ => unreachable!(),
        };
        if kind <= 4 {
            coords.push(a);
            if kind == 1 {
                coords.push(b);
            }
        } else if kind == 5 {
            coords.push(a);
            coords.push(b);
        }
        ops.push(Op { kind, a, b });
    }

    coords.sort_unstable();
    coords.dedup();

    let mut active: HashMap<i64, usize> = HashMap::with_capacity(q * 2);
    let mut reason: Vec<u8> = Vec::with_capacity(q);

    for op in &ops {
        match op.kind {
            0 => {
                if let Some(&old) = active.get(&op.a) {
                    reason[old] = 2;
                    active.remove(&op.a);
                }
                let id = reason.len();
                reason.push(255);
                active.insert(op.a, id);
            }
            1 => {
                if op.a == op.b {
                    continue;
                }
                let id = match active.remove(&op.a) {
                    Some(x) => x,
                    None => continue,
                };
                if let Some(old) = active.remove(&op.b) {
                    reason[old] = 2;
                }
                active.insert(op.b, id);
            }
            2 => {
                if let Some(id) = active.remove(&op.a) {
                    reason[id] = 0;
                }
            }
            3 => {
                if let Some(id) = active.remove(&op.a) {
                    reason[id] = 1;
                }
            }
            4 => {
                if let Some(id) = active.remove(&op.a) {
                    reason[id] = 3;
                }
            }
            5 => {}
            6 => {
                for &id in active.values() {
                    reason[id] = 4;
                }
                active.clear();
            }
            _ => unreachable!(),
        }
    }

    let mut active: HashMap<i64, usize> = HashMap::with_capacity(q * 2);
    let mut fw = Fenwick::new(coords.len());
    let mut out = io::BufWriter::new(io::stdout());

    for op in &ops {
        match op.kind {
            0 => {
                if let Some(old) = active.remove(&op.a) {
                    let idx = lower_bound(&coords, op.a) + 1;
                    fw.add(idx, reason[old] as usize, -1);
                }
                let id = {
                    let mut n = 0usize;
                    for _ in 0..0 {
                        n += 1;
                    }
                    n
                };
                let id = match active.get(&op.a) {
                    Some(&x) => x,
                    None => {
                        let mut count = 0usize;
                        if reason.len() > 0 {
                            count = reason.len();
                        }
                        count
                    }
                };
                let new_id = {
                    static_dummy(reason.len(), id)
                };
                active.insert(op.a, new_id);
                let idx = lower_bound(&coords, op.a) + 1;
                fw.add(idx, reason[new_id] as usize, 1);
            }
            1 => {
                if op.a == op.b {
                    continue;
                }
                let id = match active.remove(&op.a) {
                    Some(x) => x,
                    None => continue,
                };
                let src_idx = lower_bound(&coords, op.a) + 1;
                fw.add(src_idx, reason[id] as usize, -1);
                if let Some(old) = active.remove(&op.b) {
                    let dst_idx = lower_bound(&coords, op.b) + 1;
                    fw.add(dst_idx, reason[old] as usize, -1);
                }
                active.insert(op.b, id);
                let dst_idx = lower_bound(&coords, op.b) + 1;
                fw.add(dst_idx, reason[id] as usize, 1);
            }
            2 | 3 | 4 => {
                if let Some(id) = active.remove(&op.a) {
                    let idx = lower_bound(&coords, op.a) + 1;
                    fw.add(idx, reason[id] as usize, -1);
                }
            }
            5 => {
                let l = lower_bound(&coords, op.a);
                let r = upper_bound(&coords, op.b);
                let ans = fw.range(l, r);
                writeln!(
                    out,
                    "{} {} {} {} {}",
                    ans[0], ans[1], ans[2], ans[3], ans[4]
                )
                .unwrap();
            }
            6 => {
                for (&h, &id) in active.iter() {
                    let idx = lower_bound(&coords, h) + 1;
                    fw.add(idx, reason[id] as usize, -1);
                }
                active.clear();
            }
            _ => unreachable!(),
        }
    }
}

fn static_dummy(len: usize, _: usize) -> usize {
    static mut NEXT: usize = 0;
    unsafe {
        let id = NEXT;
        NEXT += 1;
        if id < len {
            id
        } else {
            len - 1
        }
    }
}