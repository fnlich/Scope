use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};

struct Node {
    path: String,
    parent: Option<usize>,
    children: BTreeMap<String, usize>,
    version: i64,
    ephemeral: bool,
    owner: i64,
    counter: u64,
    alive: bool,
}

struct Watch {
    owner: i64,
    kind: u8,
    active: bool,
}

fn fire(
    map: &mut HashMap<String, Vec<usize>>,
    key: &str,
    watches: &mut Vec<Watch>,
    fired: &mut Vec<usize>,
) {
    if let Some(ids) = map.remove(key) {
        for id in ids {
            if watches[id].active {
                watches[id].active = false;
                fired.push(id);
            }
        }
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let q: usize = it.next().unwrap().parse().unwrap();

    let mut nodes = Vec::<Node>::new();
    nodes.push(Node {
        path: "/".to_string(),
        parent: None,
        children: BTreeMap::new(),
        version: 0,
        ephemeral: false,
        owner: 0,
        counter: 0,
        alive: true,
    });

    let mut paths: HashMap<String, usize> = HashMap::new();
    paths.insert("/".to_string(), 0);

    let mut sessions: HashMap<i64, bool> = HashMap::new();
    let mut ephemeral_nodes: HashMap<i64, Vec<usize>> = HashMap::new();

    let mut watches = Vec::<Watch>::new();
    watches.push(Watch {
        owner: 0,
        kind: 0,
        active: false,
    });

    let mut e_watch: HashMap<String, Vec<usize>> = HashMap::new();
    let mut d_watch: HashMap<String, Vec<usize>> = HashMap::new();
    let mut c_watch: HashMap<String, Vec<usize>> = HashMap::new();

    let mut out = io::BufWriter::new(io::stdout());

    for cmd_idx in 1..=q {
        let cmd = it.next().unwrap();

        match cmd {
            "OPEN" => {
                let s: i64 = it.next().unwrap().parse().unwrap();
                if sessions.contains_key(&s) {
                    writeln!(out, "SESSIONEXISTS 0").unwrap();
                } else {
                    sessions.insert(s, true);
                    ephemeral_nodes.insert(s, Vec::new());
                    writeln!(out, "OK 0").unwrap();
                }
            }

            "WATCH" => {
                let s: i64 = it.next().unwrap().parse().unwrap();
                let k = it.next().unwrap().as_bytes()[0];
                let p = it.next().unwrap().to_string();

                match sessions.get(&s) {
                    Some(true) => {}
                    _ => {
                        writeln!(out, "NOSESSION 0").unwrap();
                        continue;
                    }
                }

                if k != b'E' && !paths.contains_key(&p) {
                    writeln!(out, "NONODE 0").unwrap();
                    continue;
                }

                let id = cmd_idx;
                watches.push(Watch {
                    owner: s,
                    kind: k,
                    active: true,
                });

                match k {
                    b'E' => e_watch.entry(p).or_default().push(id),
                    b'D' => d_watch.entry(p).or_default().push(id),
                    b'C' => c_watch.entry(p).or_default().push(id),
                    _ => unreachable!(),
                }

                writeln!(out, "OK 0").unwrap();
            }

            "CREATE" => {
                let s: i64 = it.next().unwrap().parse().unwrap();
                let parent_path = it.next().unwrap().to_string();
                let typ = it.next().unwrap().as_bytes()[0];
                let naming = it.next().unwrap().as_bytes()[0];

                match sessions.get(&s) {
                    Some(true) => {}
                    _ => {
                        writeln!(out, "NOSESSION 0").unwrap();
                        continue;
                    }
                }

                let parent_id = match paths.get(&parent_path).copied() {
                    Some(x) => x,
                    None => {
                        writeln!(out, "NONODE 0").unwrap();
                        continue;
                    }
                };

                if nodes[parent_id].ephemeral {
                    writeln!(out, "EPHEMERALPARENT 0").unwrap();
                    continue;
                }

                let final_path;
                let child_name;

                if naming == b'N' {
                    child_name = parent_path.clone();
                    unreachable!();
                } else {
                    let mut counter = nodes[parent_id].counter;
                    loop {
                        let name = format!("{:010}", counter);
                        let candidate = if parent_path == "/" {
                            format!("/{}", name)
                        } else {
                            format!("{}/{}", parent_path, name)
                        };
                        if !paths.contains_key(&candidate) {
                            final_path = candidate;
                            child_name = name;
                            nodes[parent_id].counter = counter + 1;
                            break;
                        }
                        counter += 1;
                    }
                }

                if naming == b'N' {
                    let name = parent_path;
                    let _ = name;
                }

                if naming == b'N' {
                    let p = &nodes[parent_id].path;
                    let name = it.next();
                    if name.is_some() {
                        unreachable!();
                    }
                    let _ = p;
                }

                let _ = typ;
                let _ = s;
                let _ = child_name;
                let _ = final_path;
            }

            "SET" => {
                let s: i64 = it.next().unwrap().parse().unwrap();
                let p = it.next().unwrap().to_string();
                let v: i64 = it.next().unwrap().parse().unwrap();

                match sessions.get(&s) {
                    Some(true) => {}
                    _ => {
                        writeln!(out, "NOSESSION 0").unwrap();
                        continue;
                    }
                }

                let id = match paths.get(&p).copied() {
                    Some(x) => x,
                    None => {
                        writeln!(out, "NONODE 0").unwrap();
                        continue;
                    }
                };

                if v != -1 && v != nodes[id].version {
                    writeln!(out, "BADVERSION 0").unwrap();
                    continue;
                }

                nodes[id].version += 1;

                let mut fired = Vec::new();
                fire(&mut e_watch, &p, &mut watches, &mut fired);
                fire(&mut d_watch, &p, &mut watches, &mut fired);
                fired.sort_unstable();

                write!(out, "OK {}", fired.len()).unwrap();
                for x in fired {
                    write!(out, " {}", x).unwrap();
                }
                writeln!(out).unwrap();
            }

            "DELETE" => {
                let s: i64 = it.next().unwrap().parse().unwrap();
                let p = it.next().unwrap().to_string();
                let v: i64 = it.next().unwrap().parse().unwrap();
                let mode = it.next().unwrap().as_bytes()[0];

                match sessions.get(&s) {
                    Some(true) => {}
                    _ => {
                        writeln!(out, "NOSESSION 0").unwrap();
                        continue;
                    }
                }

                if p == "/" {
                    writeln!(out, "BADROOT 0").unwrap();
                    continue;
                }

                let id = match paths.get(&p).copied() {
                    Some(x) => x,
                    None => {
                        writeln!(out, "NONODE 0").unwrap();
                        continue;
                    }
                };

                if v != -1 && v != nodes[id].version {
                    writeln!(out, "BADVERSION 0").unwrap();
                    continue;
                }

                if mode == b'N' && !nodes[id].children.is_empty() {
                    writeln!(out, "NOTEMPTY 0").unwrap();
                    continue;
                }

                let mut order = Vec::new();

                if mode == b'N' {
                    order.push(id);
                } else {
                    let mut stack = vec![(id, false)];
                    while let Some((cur, done)) = stack.pop() {
                        if done {
                            order.push(cur);
                        } else {
                            stack.push((cur, true));
                            for &child in nodes[cur].children.values().rev() {
                                stack.push((child, false));
                            }
                        }
                    }
                }

                let mut all_fired = Vec::new();

                for cur in order {
                    let path = nodes[cur].path.clone();
                    let parent = nodes[cur].parent.unwrap();
                    let parent_path = nodes[parent].path.clone();

                    let mut fired = Vec::new();
                    fire(&mut e_watch, &path, &mut watches, &mut fired);
                    fire(&mut d_watch, &path, &mut watches, &mut fired);
                    fire(&mut c_watch, &parent_path, &mut watches, &mut fired);
                    fired.sort_unstable();
                    all_fired.extend(fired);

                    let child_name = path.rsplit('/').next().unwrap().to_string();
                    nodes[parent].children.remove(&child_name);
                    paths.remove(&path);
                    nodes[cur].alive = false;
                }

                write!(out, "OK {}", all_fired.len()).unwrap();
                for x in all_fired {
                    write!(out, " {}", x).unwrap();
                }
                writeln!(out).unwrap();
            }

            "CLOSE" => {
                let s: i64 = it.next().unwrap().parse().unwrap();

                match sessions.get(&s) {
                    Some(true) => {}
                    _ => {
                        writeln!(out, "NOSESSION 0").unwrap();
                        continue;
                    }
                }

                sessions.insert(s, false);

                for w in watches.iter_mut() {
                    if w.active && w.owner == s {
                        w.active = false;
                    }
                }

                let mut ids = Vec::new();
                if let Some(v) = ephemeral_nodes.get(&s) {
                    for &id in v {
                        if nodes[id].alive {
                            ids.push(id);
                        }
                    }
                }

                ids.sort_by(|&a, &b| nodes[a].path.cmp(&nodes[b].path));

                let mut all_fired = Vec::new();

                for cur in ids {
                    if !nodes[cur].alive {
                        continue;
                    }

                    let path = nodes[cur].path.clone();
                    let parent = nodes[cur].parent.unwrap();
                    let parent_path = nodes[parent].path.clone();

                    let mut fired = Vec::new();
                    fire(&mut e_watch, &path, &mut watches, &mut fired);
                    fire(&mut d_watch, &path, &mut watches, &mut fired);
                    fire(&mut c_watch, &parent_path, &mut watches, &mut fired);
                    fired.sort_unstable();
                    all_fired.extend(fired);

                    let child_name = path.rsplit('/').next().unwrap().to_string();
                    nodes[parent].children.remove(&child_name);
                    paths.remove(&path);
                    nodes[cur].alive = false;
                }

                write!(out, "OK {}", all_fired.len()).unwrap();
                for x in all_fired {
                    write!(out, " {}", x).unwrap();
                }
                writeln!(out).unwrap();
            }

            _ => unreachable!(),
        }
    }
}