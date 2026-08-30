use std::io::{Read, Write};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
enum Param<'a> { S(&'a [u8]), I(i64) }

fn pu(s: &[u8]) -> usize { let mut v = 0usize; for &c in s { v = v * 10 + (c - b'0') as usize; } v }

fn allnum(s: &[u8]) -> bool { !s.is_empty() && s.iter().all(|&c| c >= b'0' && c <= b'9') }

fn to_int(s: &[u8]) -> Option<i64> {
    if s.is_empty() { return None; }
    let mut i = 0usize;
    let mut neg = false;
    if s[0] == b'+' { i = 1; } else if s[0] == b'-' { i = 1; neg = true; }
    if i >= s.len() { return None; }
    let mut acc: i128 = 0;
    while i < s.len() {
        let c = s[i];
        if c < b'0' || c > b'9' { return None; }
        acc = acc * 10 + (c - b'0') as i128;
        if acc > 9223372036854775808i128 { return None; }
        i += 1;
    }
    if neg {
        if acc > 9223372036854775808i128 { None } else { Some((-acc) as i64) }
    } else {
        if acc > 9223372036854775807i128 { None } else { Some(acc as i64) }
    }
}

fn bundle_valid(t: &[u8], short: &[i32; 256], kinds: &[u8]) -> bool {
    if t.len() < 2 { return false; }
    let ch = &t[1..];
    if ch.is_empty() { return false; }
    let mut i = 0usize;
    while i < ch.len() {
        let id = short[ch[i] as usize];
        if id < 0 { return false; }
        if kinds[id as usize] != b'F' { return true; }
        i += 1;
    }
    true
}

fn is_stop<'a>(t: &[u8], stops: &HashSet<&'a [u8]>) -> bool {
    if t == b"--" { return true; }
    stops.contains(t)
}

fn is_recognized<'a>(t: &[u8], map: &HashMap<&'a [u8], usize>, short: &[i32; 256], kinds: &[u8]) -> bool {
    if t.len() > 2 && t[0] == b'-' && t[1] == b'-' {
        let body = &t[2..];
        let name = match body.iter().position(|&c| c == b'=') { Some(p) => &body[..p], None => body };
        return map.contains_key(name);
    }
    if t.len() >= 2 && t[0] == b'-' && t[1] != b'-' {
        return bundle_valid(t, short, kinds);
    }
    false
}

fn take_params<'a>(
    id: usize,
    oi: usize,
    attached: Option<&'a [u8]>,
    kinds: &[u8],
    names: &[&'a [u8]],
    args: &[&'a [u8]],
    stops: &HashSet<&'a [u8]>,
    map: &HashMap<&'a [u8], usize>,
    short: &[i32; 256],
) -> Result<(Vec<Param<'a>>, usize), String> {
    let nm = String::from_utf8_lossy(names[id]).into_owned();
    let k = kinds[id];
    if k == b'F' {
        if attached.is_some() { return Err(format!("ERR FORM {} {}", nm, oi + 1)); }
        return Ok((Vec::new(), oi + 1));
    }
    let first: &'a [u8];
    let first_idx: usize;
    let mut p: usize;
    match attached {
        Some(v) => {
            if v.is_empty() { return Err(format!("ERR FORM {} {}", nm, oi + 1)); }
            first = v; first_idx = oi; p = oi + 1;
        }
        None => {
            let j = oi + 1;
            if j >= args.len() || is_stop(args[j], stops) || is_recognized(args[j], map, short, kinds) {
                return Err(format!("ERR MISSING {} {}", nm, oi + 1));
            }
            first = args[j]; first_idx = j; p = j + 1;
        }
    }
    if k == b'T' {
        return Ok((vec![Param::S(first)], p));
    }
    let v0 = match to_int(first) {
        Some(v) => v,
        None => {
            return Err(format!("ERR BADINT {} {} {}", nm, first_idx + 1, String::from_utf8_lossy(first)));
        }
    };
    if k == b'I' {
        return Ok((vec![Param::I(v0)], p));
    }
    let mut ps = vec![Param::I(v0)];
    while p < args.len() {
        let t = args[p];
        if is_stop(t, stops) { break; }
        match to_int(t) { Some(v) => { ps.push(Param::I(v)); p += 1; } None => break }
    }
    Ok((ps, p))
}

fn main() {
    let mut data = Vec::new();
    std::io::stdin().read_to_end(&mut data).unwrap();
    let mut toks: Vec<&[u8]> = Vec::new();
    {
        let mut i = 0usize;
        let n = data.len();
        while i < n {
            while i < n && data[i] <= b' ' { i += 1; }
            if i >= n { break; }
            let s = i;
            while i < n && data[i] > b' ' { i += 1; }
            toks.push(&data[s..i]);
        }
    }
    let out = std::io::stdout();
    let mut w = std::io::BufWriter::new(out.lock());
    if toks.is_empty() { return; }
    let mut pos = 0usize;
    let n = pu(toks[pos]); pos += 1;
    let s_cnt = pu(toks[pos]); pos += 1;
    let c_cnt = pu(toks[pos]); pos += 1;

    let mut names: Vec<&[u8]> = Vec::with_capacity(n);
    let mut kinds: Vec<u8> = Vec::with_capacity(n);
    let mut reps: Vec<bool> = Vec::with_capacity(n);
    let mut short = [-1i32; 256];
    let mut map: HashMap<&[u8], usize> = HashMap::with_capacity(n * 2 + 4);
    for idx in 0..n {
        let lg = toks[pos]; pos += 1;
        let sh = toks[pos]; pos += 1;
        let kd = toks[pos][0]; pos += 1;
        let rp = toks[pos][0] == b'1'; pos += 1;
        names.push(lg);
        kinds.push(kd);
        reps.push(rp);
        map.insert(lg, idx);
        if !(sh.len() == 1 && sh[0] == b'-') {
            short[sh[0] as usize] = idx as i32;
        }
    }
    let mut stops: HashSet<&[u8]> = HashSet::with_capacity(s_cnt * 2 + 4);
    for _ in 0..s_cnt { stops.insert(toks[pos]); pos += 1; }

    let mut rules: Vec<(u8, usize, Vec<usize>)> = Vec::with_capacity(c_cnt);
    for _ in 0..c_cnt {
        let rt = toks[pos][0]; pos += 1;
        if rt == b'D' {
            let x = *map.get(toks[pos]).unwrap(); pos += 1;
            let k = pu(toks[pos]); pos += 1;
            let mut l = Vec::with_capacity(k);
            for _ in 0..k { l.push(*map.get(toks[pos]).unwrap()); pos += 1; }
            rules.push((b'D', x, l));
        } else {
            let k = pu(toks[pos]); pos += 1;
            let mut l = Vec::with_capacity(k);
            for _ in 0..k { l.push(*map.get(toks[pos]).unwrap()); pos += 1; }
            rules.push((rt, 0, l));
        }
    }

    if pos < toks.len() && !allnum(toks[pos]) { pos += 1; }
    let a_cnt = if pos < toks.len() { pu(toks[pos]) } else { 0 };
    pos += 1;
    let mut args: Vec<&[u8]> = Vec::with_capacity(a_cnt);
    for _ in 0..a_cnt {
        if pos < toks.len() { args.push(toks[pos]); pos += 1; }
    }

    let mut occurred = vec![false; n];
    let mut occs: Vec<(usize, Vec<Param>)> = Vec::new();
    let mut err: Option<String> = None;
    let mut stop_at = args.len();
    let mut i = 0usize;

    'outer: while i < args.len() {
        let t = args[i];
        if is_stop(t, &stops) { stop_at = i; break; }
        if t.len() > 2 && t[0] == b'-' && t[1] == b'-' {
            let body = &t[2..];
            let eq = body.iter().position(|&c| c == b'=');
            let (name, attached) = match eq {
                Some(p) => (&body[..p], Some(&body[p + 1..])),
                None => (body, None),
            };
            let id = match map.get(name) { Some(&x) => x, None => { stop_at = i; break; } };
            if occurred[id] && !reps[id] {
                err = Some(format!("ERR DUP {} {}", String::from_utf8_lossy(names[id]), i + 1));
                break 'outer;
            }
            match take_params(id, i, attached, &kinds, &names, &args, &stops, &map, &short) {
                Ok((ps, ni)) => { occurred[id] = true; occs.push((id, ps)); i = ni; }
                Err(e) => { err = Some(e); break 'outer; }
            }
        } else if t.len() >= 2 && t[0] == b'-' && t[1] != b'-' {
            if !bundle_valid(t, &short, &kinds) { stop_at = i; break; }
            let ch = &t[1..];
            let mut j = 0usize;
            let mut ni = i + 1;
            while j < ch.len() {
                let id = short[ch[j] as usize] as usize;
                if occurred[id] && !reps[id] {
                    err = Some(format!("ERR DUP {} {}", String::from_utf8_lossy(names[id]), i + 1));
                    break 'outer;
                }
                if kinds[id] == b'F' {
                    occurred[id] = true;
                    occs.push((id, Vec::new()));
                    j += 1;
                } else {
                    let rest = &ch[j + 1..];
                    let att = if rest.is_empty() { None } else { Some(rest) };
                    match take_params(id, i, att, &kinds, &names, &args, &stops, &map, &short) {
                        Ok((ps, k2)) => { occurred[id] = true; occs.push((id, ps)); ni = k2; }
                        Err(e) => { err = Some(e); break 'outer; }
                    }
                    break;
                }
            }
            i = ni;
        } else {
            stop_at = i;
            break;
        }
    }

    if let Some(e) = err {
        writeln!(w, "{}", e).unwrap();
        return;
    }

    for (ri, r) in rules.iter().enumerate() {
        match r.0 {
            b'R' => {
                let mut ok = false;
                for &id in r.2.iter() { if occurred[id] { ok = true; break; } }
                if !ok {
                    writeln!(w, "ERR RULE {} R", ri + 1).unwrap();
                    return;
                }
            }
            b'X' => {
                let mut sel: Vec<usize> = Vec::new();
                for &id in r.2.iter() {
                    if occurred[id] { sel.push(id); if sel.len() == 2 { break; } }
                }
                if sel.len() >= 2 {
                    writeln!(w, "ERR RULE {} X {} {}", ri + 1,
                        String::from_utf8_lossy(names[sel[0]]),
                        String::from_utf8_lossy(names[sel[1]])).unwrap();
                    return;
                }
            }
            _ => {
                if occurred[r.1] {
                    for &id in r.2.iter() {
                        if !occurred[id] {
                            writeln!(w, "ERR RULE {} D {} {}", ri + 1,
                                String::from_utf8_lossy(names[r.1]),
                                String::from_utf8_lossy(names[id])).unwrap();
                            return;
                        }
                    }
                }
            }
        }
    }

    let leftn = args.len() - stop_at;
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
    buf.extend_from_slice(format!("OK {} {}\n", occs.len(), leftn).as_bytes());
    for (id, ps) in occs.iter() {
        buf.extend_from_slice(names[*id]);
        buf.extend_from_slice(format!(" {}", ps.len()).as_bytes());
        for p in ps.iter() {
            buf.push(b' ');
            match p {
                Param::S(s) => buf.extend_from_slice(s),
                Param::I(v) => buf.extend_from_slice(v.to_string().as_bytes()),
            }
        }
        buf.push(b'\n');
    }
    buf.extend_from_slice(b"LEFT");
    for k in stop_at..args.len() {
        buf.push(b' ');
        buf.extend_from_slice(args[k]);
    }
    buf.push(b'\n');
    w.write_all(&buf).unwrap();
}