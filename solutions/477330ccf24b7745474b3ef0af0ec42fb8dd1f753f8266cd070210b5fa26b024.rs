use std::io::{self, Read, Write};

fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_ascii_whitespace();
    let h0: usize = it.next().unwrap().parse().unwrap();
    let w0: usize = it.next().unwrap().parse().unwrap();
    let q: usize = it.next().unwrap().parse().unwrap();
    let k0: usize = (w0 + 63) / 64;
    let mut data: Vec<u64> = vec![0u64; h0 * k0];
    for i in 0..h0 {
        for k in 0..k0 {
            let t = it.next().unwrap();
            let v = u64::from_str_radix(t, 16).unwrap();
            data[i * k0 + k] = v;
        }
    }
    let mut rot: usize = 0;
    let mut inv: u64 = 0;

    for _ in 0..q {
        let cmd = it.next().unwrap();
        if cmd == "ROT" {
            let t: usize = it.next().unwrap().parse().unwrap();
            rot = (rot + t) % 4;
        } else if cmd == "INV" {
            inv ^= 1;
        } else {
            let r1: usize = it.next().unwrap().parse().unwrap();
            let c1: usize = it.next().unwrap().parse().unwrap();
            let r2: usize = it.next().unwrap().parse().unwrap();
            let c2: usize = it.next().unwrap().parse().unwrap();
            let b: u64 = it.next().unwrap().parse().unwrap();
            let (i1a, j1a) = cur_to_base(r1, c1, rot, h0, w0);
            let (i2a, j2a) = cur_to_base(r2, c2, rot, h0, w0);
            let ilo = if i1a < i2a { i1a } else { i2a };
            let ihi = if i1a < i2a { i2a } else { i1a };
            let jlo = if j1a < j2a { j1a } else { j2a };
            let jhi = if j1a < j2a { j2a } else { j1a };
            let v = b ^ inv;
            let wlo = jlo / 64;
            let whi = jhi / 64;
            let mut masks: Vec<u64> = Vec::with_capacity(whi - wlo + 1);
            for w in wlo..=whi {
                let start = if w == wlo { jlo % 64 } else { 0 };
                let end = if w == whi { jhi % 64 } else { 63 };
                let upper: u64 = if end == 63 {
                    u64::MAX
                } else {
                    (1u64 << (end + 1)) - 1
                };
                let lower: u64 = u64::MAX << start;
                masks.push(upper & lower);
            }
            for i in ilo..=ihi {
                let base = i * k0;
                for (idx, w) in (wlo..=whi).enumerate() {
                    let m = masks[idx];
                    if v == 1 {
                        data[base + w] |= m;
                    } else {
                        data[base + w] &= !m;
                    }
                }
            }
        }
    }

    let (fh, fw) = if rot % 2 == 0 { (h0, w0) } else { (w0, h0) };
    let fk = (fw + 63) / 64;
    let out = io::stdout();
    let mut o = io::BufWriter::new(out.lock());
    let mut buf = String::with_capacity(fh * fk * 17 + 32);
    buf.push_str(&format!("{} {}\n", fh, fw));
    let mut row: Vec<u64> = vec![0u64; fk];
    for a in 0..fh {
        for x in row.iter_mut() {
            *x = 0;
        }
        for b in 0..fw {
            let (i, j) = cur_to_base(a, b, rot, h0, w0);
            let bit = (data[i * k0 + j / 64] >> (j % 64)) & 1;
            let val = bit ^ inv;
            if val == 1 {
                row[b / 64] |= 1u64 << (b % 64);
            }
        }
        for k in 0..fk {
            buf.push_str(&format!("{:016X}", row[k]));
            if k + 1 < fk {
                buf.push(' ');
            }
        }
        buf.push('\n');
    }
    o.write_all(buf.as_bytes()).unwrap();
}

fn cur_to_base(a: usize, b: usize, rot: usize, h0: usize, w0: usize) -> (usize, usize) {
    match rot {
        0 => (a, b),
        1 => (h0 - 1 - b, a),
        2 => (h0 - 1 - a, w0 - 1 - b),
        _ => (b, w0 - 1 - a),
    }
}