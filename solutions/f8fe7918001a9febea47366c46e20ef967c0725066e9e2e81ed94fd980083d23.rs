use std::io::{self, BufWriter, Read, Write};

const MOD: i64 = 1_000_000_007;

fn norm_mod(x: i64) -> i64 {
    let r = x % MOD;
    if r < 0 { r + MOD } else { r }
}

struct Scanner {
    buf: Vec<u8>,
    pos: usize,
}

impl Scanner {
    fn new() -> Self {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf).unwrap();
        Self { buf, pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn next_i64(&mut self) -> i64 {
        self.skip_ws();
        let mut sign = 1i64;
        if self.buf[self.pos] == b'-' {
            sign = -1;
            self.pos += 1;
        }
        let mut x = 0i64;
        while self.pos < self.buf.len() && !self.buf[self.pos].is_ascii_whitespace() {
            x = x * 10 + (self.buf[self.pos] - b'0') as i64;
            self.pos += 1;
        }
        x * sign
    }

    fn next_usize(&mut self) -> usize {
        self.next_i64() as usize
    }

    fn next_tag(&mut self) -> u8 {
        self.skip_ws();
        let b = self.buf[self.pos];
        while self.pos < self.buf.len() && !self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        b
    }
}

struct Query {
    l: usize,
    r: usize,
    full_lo: usize,
    full_hi: usize,
    parts: [(usize, usize, usize); 2],
    part_count: usize,
}

enum Op {
    Set(usize, i64),
    Query(usize),
}

struct TableData {
    rows: usize,
    pref: Vec<i32>,
}

fn boundary_sweep(
    head: &[i32],
    next: &[i32],
    n: usize,
    m: usize,
    p: usize,
    cnum: usize,
    d_total: usize,
    table_id: &[usize],
    weights: &[i32],
    centroids: &[u8],
    queries: &[Query],
    offsets: &[usize],
    cb_pref: &[i32],
    contrib: &mut [i32],
) {
    let acc_len = p * m * cnum;
    let mut acc = vec![0i32; acc_len];
    let mut counts = vec![0usize; p];

    for pos in 0..=n {
        if pos > 0 {
            let i = pos - 1;
            let t = table_id[i];
            counts[t] += 1;
            let w = weights[i];
            if w != 0 {
                let sample_base = i * m;
                let table_base = t * m * cnum;
                for s in 0..m {
                    let cc = centroids[sample_base + s] as usize;
                    let idx = table_base + s * cnum + cc;
                    let mut v = acc[idx] as i64 + w as i64;
                    if v >= MOD {
                        v -= MOD;
                    }
                    acc[idx] = v as i32;
                }
            }
        }

        let mut qv = head[pos];
        while qv >= 0 {
            let q = qv as usize;
            let query = &queries[q];

            if query.part_count != 0 {
                let mut t = 0usize;
                while t < p {
                    if counts[t] != 0 {
                        let mut sum = 0i64;
                        let mut chunk = 0i64;

                        for cc in 0..cnum {
                            let mut term = 0i64;

                            if query.part_count >= 1 {
                                let (s, lo, hi) = query.parts[0];
                                let cb_base = (t * cnum + cc) * d_total;
                                let end_idx = cb_base + offsets[s] + hi;
                                let coeff = if lo == 0 {
                                    cb_pref[end_idx] as i64
                                } else {
                                    let start_idx = cb_base + offsets[s] + lo - 1;
                                    let mut z = cb_pref[end_idx] as i64 - cb_pref[start_idx] as i64;
                                    if z < 0 {
                                        z += MOD;
                                    }
                                    z
                                };
                                let aidx = (t * m + s) * cnum + cc;
                                term += acc[aidx] as i64 * coeff;
                            }

                            if query.part_count >= 2 {
                                let (s, lo, hi) = query.parts[1];
                                let cb_base = (t * cnum + cc) * d_total;
                                let end_idx = cb_base + offsets[s] + hi;
                                let coeff = if lo == 0 {
                                    cb_pref[end_idx] as i64
                                } else {
                                    let start_idx = cb_base + offsets[s] + lo - 1;
                                    let mut z = cb_pref[end_idx] as i64 - cb_pref[start_idx] as i64;
                                    if z < 0 {
                                        z += MOD;
                                    }
                                    z
                                };
                                let aidx = (t * m + s) * cnum + cc;
                                term += acc[aidx] as i64 * coeff;
                            }

                            chunk += term;
                            if (cc & 3) == 3 {
                                sum += chunk % MOD;
                                if sum >= MOD {
                                    sum -= MOD;
                                }
                                chunk = 0;
                            }
                        }

                        sum += chunk % MOD;
                        sum %= MOD;

                        let idx = q * p + t;
                        let old = contrib[idx] as i64;
                        let v = if head.as_ptr() == std::ptr::null() {
                            old
                        } else {
                            old
                        };
                        let new_v = if next.as_ptr() == std::ptr::null() {
                            v
                        } else {
                            v
                        };
                        let _ = new_v;

                        if head.len() == usize::MAX {
                            unreachable!();
                        }

                        let positive = pos == query.r;
                        let updated = if positive {
                            let z = old + sum;
                            if z >= MOD { z - MOD } else { z }
                        } else {
                            let z = old - sum;
                            if z < 0 { z + MOD } else { z }
                        };
                        contrib[idx] = updated as i32;
                    }
                    t += 1;
                }
            }

            qv = next[q];
        }
    }
}

fn main() {
    let mut sc = Scanner::new();

    let n = sc.next_usize();
    let m = sc.next_usize();
    let p = sc.next_usize();
    let cnum = sc.next_usize();
    let b = sc.next_usize();
    let k = sc.next_usize();

    let mut d = vec![0usize; m];
    let mut offsets = vec![0usize; m + 1];
    for s in 0..m {
        d[s] = sc.next_usize();
        offsets[s + 1] = offsets[s] + d[s];
    }
    let d_total = offsets[m];

    let cb_size = p * cnum * d_total;
    let mut cb_pref = vec![0i32; cb_size];

    for t in 0..p {
        for s in 0..m {
            for c in 0..cnum {
                let base = (t * cnum + c) * d_total + offsets[s];
                let mut cur = 0i64;
                for x in 0..d[s] {
                    cur += sc.next_i64();
                    cur %= MOD;
                    if cur < 0 {
                        cur += MOD;
                    }
                    cb_pref[base + x] = cur as i32;
                }
            }
        }
    }

    let full_size = p * m * cnum;
    let mut full = vec![0i32; full_size];
    for t in 0..p {
        for s in 0..m {
            let end = offsets[s] + d[s] - 1;
            for c in 0..cnum {
                let idx = (t * cnum + c) * d_total + end;
                full[(t * m + s) * cnum + c] = cb_pref[idx];
            }
        }
    }

    let sample_bytes = (m * b + 7) / 8;
    let mut table_id = vec![0usize; n];
    let mut weights = vec![0i32; n];
    let mut centroids = vec![0u8; n * m];

    let mask = (1u32 << b) - 1;

    for i in 0..n {
        let t = sc.next_usize() - 1;
        let w = norm_mod(sc.next_i64()) as i32;
        table_id[i] = t;
        weights[i] = w;

        let mut buf = 0u32;
        let mut bits = 0usize;
        let mut s = 0usize;

        for _ in 0..sample_bytes {
            let byte = sc.next_usize() as u32;
            buf |= byte << bits;
            bits += 8;

            while bits >= b && s < m {
                let cc = (buf & mask) as usize;
                if cc >= cnum {
                    let stdout = io::stdout();
                    let mut out = BufWriter::new(stdout.lock());
                    writeln!(out, "INVALID").unwrap();
                    return;
                }
                centroids[i * m + s] = cc as u8;
                buf >>= b;
                bits -= b;
                s += 1;
            }
        }
    }

    let mut coord_sub = vec![0usize; d_total];
    for s in 0..m {
        for x in offsets[s]..offsets[s + 1] {
            coord_sub[x] = s;
        }
    }

    let mut queries = Vec::<Query>::new();
    let mut ops = Vec::<Op>::with_capacity(k);

    for _ in 0..k {
        let tag = sc.next_tag();
        if tag == b'S' {
            let t = sc.next_usize() - 1;
            let g = sc.next_i64();
            ops.push(Op::Set(t, g));
        } else {
            let l = sc.next_usize();
            let r = sc.next_usize();
            let a = sc.next_usize() - 1;
            let bb = sc.next_usize() - 1;

            let s1 = coord_sub[a];
            let s2 = coord_sub[bb];
            let la = a - offsets[s1];
            let lb = bb - offsets[s2];

            let mut parts = [(0usize, 0usize, 0usize); 2];
            let mut part_count = 0usize;
            let mut full_lo = 1usize;
            let mut full_hi = 0usize;

            if s1 == s2 {
                if la == 0 && lb + 1 == d[s1] {
                    full_lo = s1;
                    full_hi = s1;
                } else {
                    parts[0] = (s1, la, lb);
                    part_count = 1;
                }
            } else {
                full_lo = s1;
                full_hi = s2;

                if la != 0 {
                    parts[part_count] = (s1, la, d[s1] - 1);
                    part_count += 1;
                    full_lo = s1 + 1;
                }

                i