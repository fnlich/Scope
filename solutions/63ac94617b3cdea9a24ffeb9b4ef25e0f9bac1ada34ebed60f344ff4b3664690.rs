use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::rc::Rc;

type Seg = (i64, i64, i64);
type Row = Rc<Vec<Seg>>;

struct Scanner {
    buf: Vec<u8>,
    pos: usize,
}

impl Scanner {
    fn new() -> Scanner {
        let mut b = Vec::new();
        std::io::stdin().read_to_end(&mut b).unwrap();
        Scanner { buf: b, pos: 0 }
    }
    fn skip_ws(&mut self) {
        while self.pos < self.buf.len() {
            let c = self.buf[self.pos];
            if c == b' ' || (c >= 0x09 && c <= 0x0D) {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn token(&mut self) -> &[u8] {
        self.skip_ws();
        let s = self.pos;
        while self.pos < self.buf.len() {
            let c = self.buf[self.pos];
            if c == b' ' || (c >= 0x09 && c <= 0x0D) {
                break;
            }
            self.pos += 1;
        }
        &self.buf[s..self.pos]
    }
    fn next_i64(&mut self) -> i64 {
        self.skip_ws();
        let mut neg = false;
        if self.pos < self.buf.len() && (self.buf[self.pos] == b'-' || self.buf[self.pos] == b'+') {
            neg = self.buf[self.pos] == b'-';
            self.pos += 1;
        }
        let mut v: i128 = 0;
        while self.pos < self.buf.len() {
            let c = self.buf[self.pos];
            if c >= b'0' && c <= b'9' {
                v = v * 10 + (c - b'0') as i128;
                self.pos += 1;
            } else {
                break;
            }
        }
        if neg {
            v = -v;
        }
        v as i64
    }
}

fn value_at(row: &[Seg], x: i64) -> (i64, i64) {
    let mut lo = 0usize;
    let mut hi = row.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if row[mid].0 <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let idx = lo - 1;
    (row[idx].1, row[idx].2)
}

fn splice(dst: &[Seg], x1: i64, x2: i64, mid: &[Seg], w: i64) -> Vec<Seg> {
    let mut out: Vec<Seg> = Vec::with_capacity(dst.len() + mid.len() + 2);
    for &s in dst.iter() {
        if s.0 < x1 {
            out.push(s);
        } else {
            break;
        }
    }
    for &m in mid.iter() {
        out.push(m);
    }
    if x2 < w {
        let v = value_at(dst, x2);
        out.push((x2, v.0, v.1));
        for &s in dst.iter() {
            if s.0 > x2 {
                out.push(s);
            }
        }
    }
    let mut res: Vec<Seg> = Vec::with_capacity(out.len());
    for s in out.into_iter() {
        let mut skip = false;
        if let Some(l) = res.last() {
            if l.1 == s.1 && l.2 == s.2 {
                skip = true;
            }
        }
        if !skip {
            res.push(s);
        }
    }
    res
}

fn extract(src: &[Seg], x1: i64, x2: i64) -> Vec<Seg> {
    let mut mid: Vec<Seg> = Vec::new();
    let v = value_at(src, x1);
    mid.push((x1, v.0, v.1));
    for &s in src.iter() {
        if s.0 > x1 && s.0 < x2 {
            mid.push(s);
        }
    }
    mid
}

fn rows_equal(a: &Row, b: &Row) -> bool {
    if Rc::ptr_eq(a, b) {
        return true;
    }
    a.as_ref() == b.as_ref()
}

fn main() {
    let mut sc = Scanner::new();
    let out_handle = std::io::stdout();
    let mut out = std::io::BufWriter::new(out_handle.lock());

    let h = sc.next_i64();
    let w = sc.next_i64();
    let b = sc.next_i64();

    let default_row: Row = Rc::new(vec![(0i64, 0i64, 0i64)]);
    let hs = h as usize;
    let mut rows: Vec<Row> = vec![default_row.clone(); hs];

    for _ in 0..b {
        let q = sc.next_i64();
        let mut opening: BTreeMap<usize, Row> = BTreeMap::new();
        for _ in 0..q {
            let t = sc.token();
            let c = if t.is_empty() { b'?' } else { t[0] };
            if c == b'F' {
                let y1 = sc.next_i64();
                let y2 = sc.next_i64();
                let x1 = sc.next_i64();
                let x2 = sc.next_i64();
                let g = sc.next_i64();
                let a = sc.next_i64();
                let cy1 = if y1 < 0 { 0 } else { y1 };
                let cy2 = if y2 > h { h } else { y2 };
                let cx1 = if x1 < 0 { 0 } else { x1 };
                let cx2 = if x2 > w { w } else { x2 };
                if cy1 >= cy2 || cx1 >= cx2 {
                    continue;
                }
                let mid = vec![(cx1, g, a)];
                let full = cx1 == 0 && cx2 == w;
                let full_row: Row = if full {
                    Rc::new(vec![(0, g, a)])
                } else {
                    default_row.clone()
                };
                let mut y = cy1;
                while y < cy2 {
                    let idx = y as usize;
                    if !opening.contains_key(&idx) {
                        opening.insert(idx, rows[idx].clone());
                    }
                    if full {
                        rows[idx] = full_row.clone();
                    } else {
                        let nv = splice(rows[idx].as_ref(), cx1, cx2, &mid, w);
                        rows[idx] = Rc::new(nv);
                    }
                    y += 1;
                }
            } else if c == b'L' {
                let y = sc.next_i64();
                let x = sc.next_i64();
                let k = sc.next_i64();
                let mut pairs: Vec<(i64, i64)> = Vec::with_capacity(k as usize);
                for _ in 0..k {
                    let g = sc.next_i64();
                    let a = sc.next_i64();
                    pairs.push((g, a));
                }
                if y < 0 || y >= h || k <= 0 {
                    continue;
                }
                let xend: i128 = x as i128 + k as i128;
                let cx1: i128 = if (x as i128) < 0 { 0 } else { x as i128 };
                let cx2: i128 = if xend > w as i128 { w as i128 } else { xend };
                if cx1 >= cx2 {
                    continue;
                }
                let cx1 = cx1 as i64;
                let cx2 = cx2 as i64;
                let mut mid: Vec<Seg> = Vec::new();
                let mut j = cx1;
                while j < cx2 {
                    let off = (j as i128 - x as i128) as usize;
                    let p = pairs[off];
                    let mut skip = false;
                    if let Some(l) = mid.last() {
                        if l.1 == p.0 && l.2 == p.1 {
                            skip = true;
                        }
                    }
                    if !skip {
                        mid.push((j, p.0, p.1));
                    }
                    j += 1;
                }
                let idx = y as usize;
                if !opening.contains_key(&idx) {
                    opening.insert(idx, rows[idx].clone());
                }
                let nv = splice(rows[idx].as_ref(), cx1, cx2, &mid, w);
                rows[idx] = Rc::new(nv);
            } else if c == b'R' {
                let y1 = sc.next_i64();
                let y2 = sc.next_i64();
                let x1 = sc.next_i64();
                let x2 = sc.next_i64();
                let cy1 = if y1 < 0 { 0 } else { y1 };
                let cy2 = if y2 > h { h } else { y2 };
                let cx1 = if x1 < 0 { 0 } else { x1 };
                let cx2 = if x2 > w { w } else { x2 };
                if cy1 >= cy2 || cx1 >= cx2 {
                    continue;
                }
                let full = cx1 == 0 && cx2 == w;
                let mut y = cy1;
                while y < cy2 {
                    let idx = y as usize;
                    let orig: Row = match opening.get(&idx) {
                        Some(r) => r.clone(),
                        None => rows[idx].clone(),
                    };
                    if !opening.contains_key(&idx) {
                        opening.insert(idx, rows[idx].clone());
                    }
                    if full {
                        rows[idx] = orig;
                    } else {
                        let mid = extract(orig.as_ref(), cx1, cx2);
                        let nv = splice(rows[idx].as_ref(), cx1, cx2, &mid, w);
                        rows[idx] = Rc::new(nv);
                    }
                    y += 1;
                }
            } else if c == b'C' {
                let yd = sc.next_i64();
                let ys = sc.next_i64();
                let n = sc.next_i64();
                let x1 = sc.next_i64();
                let x2 = sc.next_i64();
                let cx1 = if x1 < 0 { 0 } else { x1 };
                let cx2 = if x2 > w { w } else { x2 };
                if cx1 >= cx2 || n <= 0 {
                    continue;
                }
                let ydi = yd as i128;
                let ysi = ys as i128;
                let hi128 = h as i128;
                let mut lo: i128 = 0;
                if -ydi > lo {
                    lo = -ydi;
                }
                if -ysi > lo {
                    lo = -ysi;
                }
                let mut hi: i128 = n as i128;
                if hi128 - ydi < hi {
                    hi = hi128 - ydi;
                }
                if hi128 - ysi < hi {
                    hi = hi128 - ysi;
                }
                if lo >= hi {
                    continue;
                }
                let cnt = (hi - lo) as usize;
                let mut plan: Vec<(usize, Row)> = Vec::with_capacity(cnt);
                let mut i = lo;
                while i < hi {
                    let d = (ydi + i) as usize;
                    let s = (ysi + i) as usize;
                    plan.push((d, rows[s].clone()));
                    i += 1;
                }
                let full = cx1 == 0 && cx2 == w;
                for (d, srow) in plan.into_iter() {
                    if !opening.contains_key(&d) {
                        opening.insert(d, rows[d].clone());
                    }
                    if full {
                        rows[d] = srow;
                    } else {
                        let mid = extract(srow.as_ref(), cx1, cx2);
                        let nv = splice(rows[d].as_ref(), cx1, cx2, &mid, w);
                        rows[d] = Rc::new(nv);
                    }
                }
            }
        }
        let mut changed: Vec<usize> = Vec::new();
        for (idx, orig) in opening.iter() {
            if !rows_equal(&rows[*idx], orig) {
                changed.push(*idx);
            }
        }
        let mut line = String::new();
        line.push_str(&changed.len().to_string());
        for idx in changed.iter() {
            line.push(' ');
            line.push_str(&idx.to_string());
        }
        line.push('\n');
        out.write_all(line.as_bytes()).unwrap();
    }
    out.flush().unwrap();
}