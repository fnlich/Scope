use std::io::{Read, Write};
use std::collections::{HashMap, HashSet};

fn fail(code: &str, path: &[usize]) -> ! {
    let mut o = String::new();
    o.push_str("ERROR ");
    o.push_str(code);
    o.push(' ');
    o.push_str(&path.len().to_string());
    for x in path {
        o.push(' ');
        o.push_str(&x.to_string());
    }
    o.push('\n');
    let so = std::io::stdout();
    let mut w = so.lock();
    let _ = w.write_all(o.as_bytes());
    let _ = w.flush();
    std::process::exit(0);
}

struct Frame<'a> {
    total: usize,
    count: usize,
    keys: Vec<&'a str>,
}

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let toks: Vec<&str> = s.split_ascii_whitespace().collect();
    if toks.is_empty() {
        return;
    }
    let n: usize = toks[0].parse().unwrap();
    let mut p: usize = 1;

    let mut env: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut stack: Vec<Frame> = Vec::new();
    let mut path: Vec<usize> = Vec::new();
    let mut out = String::new();
    out.push_str("OK ");
    out.push_str(&n.to_string());
    out.push('\n');

    for _node in 0..n {
        let dl = stack.len();
        if dl > 0 {
            stack[dl - 1].count += 1;
            let c = stack[dl - 1].count;
            path.truncate(dl - 1);
            path.push(c);
        } else {
            path.clear();
        }

        let t = toks[p];
        p += 1;

        if t == "G" {
            let label = toks[p];
            p += 1;
            let d: usize = toks[p].parse().unwrap();
            p += 1;
            let mut keys: Vec<&str> = Vec::with_capacity(d);
            let mut vals: Vec<&str> = Vec::with_capacity(d);
            let mut own: HashMap<&str, usize> = HashMap::with_capacity(d * 2 + 1);
            let mut dup = false;
            for i in 0..d {
                let k = toks[p];
                p += 1;
                let v = toks[p];
                p += 1;
                keys.push(k);
                vals.push(v);
                if own.insert(k, i).is_some() {
                    dup = true;
                }
            }
            if dup {
                fail("DUP_DEFAULT", &path);
            }
            let k_children: usize = toks[p].parse().unwrap();
            p += 1;

            let mut state: Vec<u8> = vec![0u8; d];
            let mut resolved: Vec<&str> = vec![""; d];

            for i in 0..d {
                if state[i] == 2 {
                    continue;
                }
                let mut trail: Vec<usize> = Vec::new();
                let mut cur = i;
                let result: &str;
                loop {
                    state[cur] = 1;
                    trail.push(cur);
                    let v = vals[cur];
                    if v.as_bytes()[0] == b'@' {
                        let key = &v[1..];
                        if let Some(&j) = own.get(key) {
                            if state[j] == 2 {
                                result = resolved[j];
                                break;
                            } else if state[j] == 1 {
                                match env.get(key).and_then(|x| x.last().copied()) {
                                    Some(val) => {
                                        result = val;
                                        break;
                                    }
                                    None => {
                                        fail("DEFAULT_CYCLE", &path);
                                    }
                                }
                            } else {
                                cur = j;
                                continue;
                            }
                        } else {
                            match env.get(key).and_then(|x| x.last().copied()) {
                                Some(val) => {
                                    result = val;
                                    break;
                                }
                                None => {
                                    fail("UNBOUND_DEFAULT", &path);
                                }
                            }
                        }
                    } else {
                        result = v;
                        break;
                    }
                }
                for &idx in trail.iter() {
                    state[idx] = 2;
                    resolved[idx] = result;
                }
            }

            let mut pushed: Vec<&str> = Vec::with_capacity(d);
            for i in 0..d {
                env.entry(keys[i]).or_insert_with(Vec::new).push(resolved[i]);
                pushed.push(keys[i]);
            }

            out.push('G');
            out.push(' ');
            out.push_str(label);
            out.push(' ');
            out.push_str(&k_children.to_string());
            out.push('\n');

            stack.push(Frame {
                total: k_children,
                count: 0,
                keys: pushed,
            });
        } else if t == "B" {
            let key = toks[p];
            p += 1;
            let value = toks[p];
            p += 1;
            let lab = toks[p];
            p += 1;
            let rl = if lab == "-" { key } else { lab };
            out.push('B');
            out.push(' ');
            out.push_str(key);
            out.push(' ');
            out.push_str(value);
            out.push(' ');
            out.push_str(rl);
            out.push('\n');
        } else {
            let key = toks[p];
            p += 1;
            let sel = toks[p];
            p += 1;
            let o_count: usize = toks[p].parse().unwrap();
            p += 1;
            let mut names: Vec<&str> = Vec::with_capacity(o_count);
            for _ in 0..o_count {
                names.push(toks[p]);
                p += 1;
            }
            if p < toks.len() && toks[p] == "L" {
                p += 1;
            }
            let l_count: usize = toks[p].parse().unwrap();
            p += 1;

            let mut seen: HashSet<&str> = HashSet::with_capacity(o_count * 2 + 1);
            let mut dupo = false;
            for nm in names.iter() {
                if !seen.insert(*nm) {
                    dupo = true;
                }
            }
            if dupo {
                fail("DUP_OPTION", &path);
            }
            if l_count != 0 && l_count != o_count {
                fail("LABEL_COUNT", &path);
            }
            let mut labels: Vec<&str> = Vec::with_capacity(l_count);
            for _ in 0..l_count {
                labels.push(toks[p]);
                p += 1;
            }

            let effective: Option<&str> = if sel != "-" {
                Some(sel)
            } else {
                env.get(key).and_then(|x| x.last().copied())
            };

            let mut index: i64 = -1;
            if let Some(e) = effective {
                let mut found = false;
                for (i, nm) in names.iter().enumerate() {
                    if *nm == e {
                        index = i as i64;
                        found = true;
                        break;
                    }
                }
                if !found {
                    fail("UNKNOWN_SELECTION", &path);
                }
            }

            out.push('C');
            out.push(' ');
            out.push_str(key);
            out.push(' ');
            out.push_str(&index.to_string());
            out.push(' ');
            out.push_str(&o_count.to_string());
            for i in 0..o_count {
                out.push(' ');
                out.push_str(names[i]);
                out.push(' ');
                if l_count == 0 {
                    out.push_str(names[i]);
                } else {
                    out.push_str(labels[i]);
                }
            }
            out.push('\n');
        }

        loop {
            let done = match stack.last() {
                Some(f) => f.count == f.total,
                None => false,
            };
            if !done {
                break;
            }
            let fr = stack.pop().unwrap();
            for k in fr.keys {
                if let Some(v) = env.get_mut(k) {
                    v.pop();
                }
            }
        }
    }

    let so = std::io::stdout();
    let mut w = std::io::BufWriter::new(so.lock());
    let _ = w.write_all(out.as_bytes());
    let _ = w.flush();
}