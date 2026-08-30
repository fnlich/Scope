use std::io::{self, Read, Write};
use std::collections::{BTreeSet, BinaryHeap};
use std::cmp::Reverse;

struct Conn {
    q: i64,
    p: u8,
    h: i64,
    succ: i64,
    idle_time: i64,
}

fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_ascii_whitespace();
    let n: usize = it.next().unwrap().parse().unwrap();
    let c: usize = it.next().unwrap().parse().unwrap();

    let mut arr_a: Vec<i64> = vec![0; n];
    let mut arr_d: Vec<i64> = vec![0; n];
    let mut arr_b: Vec<bool> = vec![false; n];
    let mut arr_x: Vec<bool> = vec![false; n];
    let mut valid: Vec<bool> = vec![false; n];

    for i in 0..n {
        let a: i64 = it.next().unwrap().parse().unwrap();
        let d: i64 = it.next().unwrap().parse().unwrap();
        let m = it.next().unwrap();
        let l: i64 = it.next().unwrap().parse().unwrap();
        let e: i64 = it.next().unwrap().parse().unwrap();
        let x: i64 = it.next().unwrap().parse().unwrap();
        arr_a[i] = a;
        arr_d[i] = d;
        arr_b[i] = m.as_bytes()[0] == b'B';
        arr_x[i] = x == 1;
        valid[i] = !(l >= 0 && l != e);
    }

    let mut dq: Vec<i64> = Vec::with_capacity(2 * n);
    let mut dp: Vec<u8> = Vec::with_capacity(2 * n);
    let mut dh: Vec<i64> = Vec::with_capacity(2 * n);
    for _ in 0..(2 * n) {
        let q: i64 = it.next().unwrap().parse().unwrap();
        let p: i64 = it.next().unwrap().parse().unwrap();
        let h: i64 = it.next().unwrap().parse().unwrap();
        dq.push(q);
        dp.push(p as u8);
        dh.push(h);
    }

    let mut status: Vec<u8> = vec![0; n];
    let mut attempts_of: Vec<Vec<u32>> = vec![Vec::new(); n];

    let mut order: Vec<usize> = (0..n).filter(|&i| valid[i]).collect();
    order.sort_by(|&i, &j| (arr_a[i], i).cmp(&(arr_a[j], j)));
    let mut aptr: usize = 0;

    let mut conns: Vec<Conn> = Vec::new();
    let mut open: usize = 0;

    let mut idle: BTreeSet<(i64, u32)> = BTreeSet::new();
    let mut timeouts: BTreeSet<(i64, u32)> = BTreeSet::new();
    let mut waiting: BTreeSet<(i64, u32, u8)> = BTreeSet::new();

    let mut recs: Vec<(u32, u8, u32, bool)> = Vec::new();
    let mut ends: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();

    loop {
        let mut t: Option<i64> = None;
        if let Some(&Reverse((et, _))) = ends.peek() {
            t = Some(match t {
                None => et,
                Some(v) => if et < v { et } else { v },
            });
        }
        if let Some(&(ct, _)) = timeouts.iter().next() {
            t = Some(match t {
                None => ct,
                Some(v) => if ct < v { ct } else { v },
            });
        }
        if aptr < order.len() {
            let at = arr_a[order[aptr]];
            t = Some(match t {
                None => at,
                Some(v) => if at < v { at } else { v },
            });
        }
        if let Some(&(wt, _, _)) = waiting.iter().next() {
            t = Some(match t {
                None => wt,
                Some(v) => if wt < v { wt } else { v },
            });
        }
        let now = match t {
            None => break,
            Some(v) => v,
        };

        loop {
            let take = match ends.peek() {
                Some(&Reverse((et, _))) => et == now,
                None => false,
            };
            if !take {
                break;
            }
            let Reverse((_, idx)) = ends.pop().unwrap();
            let (rid, att, cid, stale) = recs[idx];
            let ri = rid as usize;
            let ci = (cid - 1) as usize;
            if stale {
                open -= 1;
                let retry = att == 0 && (arr_b[ri] || conns[ci].p == 0);
                if retry {
                    waiting.insert((now, rid, 1));
                } else {
                    status[ri] = 2;
                }
            } else {
                conns[ci].succ += 1;
                status[ri] = 1;
                if arr_x[ri] {
                    open -= 1;
                } else {
                    conns[ci].idle_time = now;
                    idle.insert((-now, cid));
                    let h = conns[ci].h;
                    timeouts.insert((now + h, cid));
                }
            }
        }

        loop {
            let first = match timeouts.iter().next() {
                Some(&(ct, cid)) => {
                    if ct == now {
                        Some((ct, cid))
                    } else {
                        None
                    }
                }
                None => None,
            };
            match first {
                Some((ct, cid)) => {
                    timeouts.remove(&(ct, cid));
                    let ci = (cid - 1) as usize;
                    idle.remove(&(-conns[ci].idle_time, cid));
                    open -= 1;
                }
                None => break,
            }
        }

        while aptr < order.len() && arr_a[order[aptr]] == now {
            let ri = order[aptr];
            waiting.insert((now, ri as u32, 0));
            aptr += 1;
        }

        loop {
            let head = match waiting.iter().next() {
                Some(&(e, rid, att)) => {
                    if e <= now {
                        Some((e, rid, att))
                    } else {
                        None
                    }
                }
                None => None,
            };
            let (e, rid, att) = match head {
                Some(v) => v,
                None => break,
            };

            let mut chosen: Option<(u32, bool)> = None;
            if let Some(&(nt, cid)) = idle.iter().next() {
                idle.remove(&(nt, cid));
                let ci = (cid - 1) as usize;
                timeouts.remove(&(conns[ci].idle_time + conns[ci].h, cid));
                chosen = Some((cid, true));
            } else if open < c {
                let di = conns.len();
                conns.push(Conn {
                    q: dq[di],
                    p: dp[di],
                    h: dh[di],
                    succ: 0,
                    idle_time: 0,
                });
                open += 1;
                chosen = Some((conns.len() as u32, false));
            }

            let (cid, reused) = match chosen {
                Some(v) => v,
                None => break,
            };

            waiting.remove(&(e, rid, att));
            let ri = rid as usize;
            let ci = (cid - 1) as usize;
            let stale = reused && conns[ci].succ >= conns[ci].q;
            attempts_of[ri].push(cid);
            let idx = recs.len();
            recs.push((rid, att, cid, stale));
            ends.push(Reverse((now + arr_d[ri], idx)));
        }
    }

    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());
    let mut buf = String::with_capacity(n * 12);
    for i in 0..n {
        match status[i] {
            0 => buf.push_str("INVALID 0\n"),
            1 => {
                buf.push_str("OK ");
                buf.push_str(&attempts_of[i].len().to_string());
                for cid in &attempts_of[i] {
                    buf.push(' ');
                    buf.push_str(&cid.to_string());
                }
                buf.push('\n');
            }
            _ => {
                buf.push_str("FAIL ");
                buf.push_str(&attempts_of[i].len().to_string());
                for cid in &attempts_of[i] {
                    buf.push(' ');
                    buf.push_str(&cid.to_string());
                }
                buf.push('\n');
            }
        }
    }
    w.write_all(buf.as_bytes()).unwrap();
}