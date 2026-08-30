use std::io::{self, Read, Write};
use std::collections::HashMap;

fn main() {
    let mut data = Vec::new();
    io::stdin().read_to_end(&mut data).unwrap();
    let mut toks: Vec<(usize, usize)> = Vec::new();
    {
        let mut i = 0usize;
        let len = data.len();
        while i < len {
            while i < len && (data[i] == b' ' || (data[i] >= 0x09 && data[i] <= 0x0D)) {
                i += 1;
            }
            if i >= len {
                break;
            }
            let s = i;
            while i < len && !(data[i] == b' ' || (data[i] >= 0x09 && data[i] <= 0x0D)) {
                i += 1;
            }
            toks.push((s, i));
        }
    }
    if toks.is_empty() {
        return;
    }
    let parse_num = |r: (usize, usize)| -> i64 {
        let mut v: i64 = 0;
        let mut neg = false;
        let mut k = r.0;
        if k < r.1 && (data[k] == b'-' || data[k] == b'+') {
            neg = data[k] == b'-';
            k += 1;
        }
        while k < r.1 {
            v = v * 10 + (data[k] - b'0') as i64;
            k += 1;
        }
        if neg {
            -v
        } else {
            v
        }
    };
    let n = parse_num(toks[0]) as usize;
    let mut kinds: Vec<u8> = Vec::with_capacity(n);
    let mut names: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut targets: Vec<i64> = Vec::with_capacity(n);
    let mut ti = 1usize;
    for _ in 0..n {
        if ti >= toks.len() {
            break;
        }
        let r = toks[ti];
        ti += 1;
        let w = &data[r.0..r.1];
        if w == b"OPEN" {
            kinds.push(0);
            names.push(toks[ti]);
            ti += 1;
            targets.push(0);
        } else if w == b"CLOSE" {
            kinds.push(1);
            names.push(toks[ti]);
            ti += 1;
            targets.push(0);
        } else {
            kinds.push(2);
            let p = toks[ti];
            ti += 1;
            let t = parse_num(toks[ti]);
            ti += 1;
            names.push(p);
            targets.push(t);
        }
    }
    let m = kinds.len();

    let mut map: HashMap<&[u8], u32> = HashMap::new();
    let mut stamp: u32 = 0;

    // returns (valid, first_failure_index_0based or m)
    let mut check = |skip: usize, map: &mut HashMap<&[u8], u32>, stamp: &mut u32| -> (bool, usize) {
        *stamp += 1;
        let st = *stamp;
        let mut active: Option<(usize, usize)> = None;
        for i in 0..m {
            if i == skip {
                continue;
            }
            match kinds[i] {
                0 => {
                    if active.is_some() {
                        return (false, i);
                    }
                    let key: &[u8] = unsafe { std::mem::transmute(&data[names[i].0..names[i].1]) };
                    let e = map.entry(key).or_insert(0);
                    if *e == st {
                        return (false, i);
                    }
                    *e = st;
                    active = Some(names[i]);
                }
                1 => {
                    let ok = match active {
                        Some(a) => {
                            data[a.0..a.1] == data[names[i].0..names[i].1]
                        }
                        None => false,
                    };
                    if !ok {
                        return (false, i);
                    }
                    active = None;
                }
                _ => {}
            }
        }
        if active.is_some() {
            return (false, m);
        }
        (true, m)
    };

    let (ok0, f) = check(usize::MAX, &mut map, &mut stamp);
    let mut chosen: usize = usize::MAX;
    if !ok0 {
        let mut fallback: usize = usize::MAX;
        let hi = if f >= m { m } else { f + 1 };
        for j in 0..hi {
            if kinds[j] == 2 {
                continue;
            }
            if fallback == usize::MAX {
                fallback = j;
            }
            let (v, _) = check(j, &mut map, &mut stamp);
            if v {
                chosen = j;
                break;
            }
        }
        if chosen == usize::MAX {
            if fallback == usize::MAX {
                for j in 0..m {
                    if kinds[j] != 2 {
                        fallback = j;
                        break;
                    }
                }
            }
            chosen = fallback;
        }
    }

    let mut out_nums: Vec<u32> = Vec::new();
    let mut out_paths: Vec<(usize, usize)> = Vec::new();
    let mut out_groups: Vec<(usize, usize)> = Vec::new();
    let mut out_gid: Vec<u32> = Vec::new();
    let mut out_target: Vec<i64> = Vec::new();

    let mut active_name: Option<(usize, usize)> = None;
    let mut active_gid: u32 = 0;
    let mut gcounter: u32 = 0;
    let mut rnum: u32 = 0;
    for i in 0..m {
        if i == chosen {
            continue;
        }
        match kinds[i] {
            0 => {
                gcounter += 1;
                active_gid = gcounter;
                active_name = Some(names[i]);
            }
            1 => {
                active_gid = 0;
                active_name = None;
            }
            _ => {
                rnum += 1;
                out_nums.push(rnum);
                out_paths.push(names[i]);
                match active_name {
                    Some(a) => out_groups.push(a),
                    None => out_groups.push((usize::MAX, usize::MAX)),
                }
                out_gid.push(active_gid);
                out_target.push(targets[i]);
            }
        }
    }

    let cnt = out_nums.len();
    let mut gid_by_num: Vec<u32> = vec![u32::MAX; cnt + 2];
    for k in 0..cnt {
        gid_by_num[out_nums[k] as usize] = out_gid[k];
    }

    let stdout = io::stdout();
    let lock = stdout.lock();
    let mut w = io::BufWriter::with_capacity(1 << 20, lock);
    let ri = if chosen == usize::MAX { 0 } else { chosen + 1 };
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    buf.extend_from_slice(ri.to_string().as_bytes());
    buf.push(b' ');
    buf.extend_from_slice(cnt.to_string().as_bytes());
    buf.push(b'\n');
    w.write_all(&buf).unwrap();
    for k in 0..cnt {
        let mut line: Vec<u8> = Vec::with_capacity(64);
        line.extend_from_slice(out_nums[k].to_string().as_bytes());
        line.extend_from_slice(b" GET ");
        line.extend_from_slice(&data[out_paths[k].0..out_paths[k].1]);
        line.push(b' ');
        if out_groups[k].0 == usize::MAX {
            line.push(b'-');
        } else {
            line.extend_from_slice(&data[out_groups[k].0..out_groups[k].1]);
        }
        line.push(b' ');
        let t = out_target[k];
        let mut sock = false;
        if t != 0 {
            let tu = t as usize;
            if tu < gid_by_num.len() && gid_by_num[tu] != u32::MAX && gid_by_num[tu] == out_gid[k] {
                sock = true;
            }
        }
        if sock {
            line.extend_from_slice(b"SOCKET");
        } else {
            line.extend_from_slice(b"PAGE");
        }
        line.push(b'\n');
        w.write_all(&line).unwrap();
    }
}