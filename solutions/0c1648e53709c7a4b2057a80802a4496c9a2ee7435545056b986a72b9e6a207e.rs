use std::collections::{BTreeSet, HashMap};
use std::io::{self, Read, Write};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_ascii_whitespace();

    let mode = it.next().unwrap().as_bytes()[0];
    let p: usize = it.next().unwrap().parse().unwrap();
    let a: usize = it.next().unwrap().parse().unwrap();
    let k: usize = it.next().unwrap().parse().unwrap();
    let d: usize = it.next().unwrap().parse().unwrap();

    let mut names = Vec::with_capacity(p + a);
    let mut ids: HashMap<String, usize> = HashMap::with_capacity(p + a);
    let mut canonical = vec![false; p + a];

    for i in 0..p {
        let s = it.next().unwrap().to_string();
        ids.insert(s.clone(), i);
        names.push(s);
        canonical[i] = true;
    }

    let mut targets: Vec<Option<usize>> = vec![None; p + a];
    let mut alias_ids = Vec::with_capacity(a);

    for _ in 0..a {
        let name = it.next().unwrap().to_string();
        let target = it.next().unwrap().to_string();
        let id = names.len();
        ids.insert(name.clone(), id);
        names.push(name);
        canonical.push(false);
        let tid = ids.get(&target).copied();
        targets.push(tid);
        alias_ids.push(id);
    }

    let mut source_key = Vec::with_capacity(k);
    let mut source_count = Vec::with_capacity(k);
    for _ in 0..k {
        let key = it.next().unwrap().to_string();
        let count: i64 = it.next().unwrap().parse().unwrap();
        source_key.push(key);
        source_count.push(count);
    }

    let mut rules = Vec::with_capacity(d);
    for _ in 0..d {
        let before = it.next().unwrap().to_string();
        let after = it.next().unwrap().to_string();
        rules.push((before, after));
    }

    let n = names.len();
    let mut state = vec![0u8; n];
    let mut resolved = vec![None; n];

    for i in 0..p {
        state[i] = 2;
        resolved[i] = Some(i);
    }

    let mut invalid_alias = vec![false; n];

    for &start in &alias_ids {
        if state[start] != 0 {
            continue;
        }

        let mut path = Vec::new();
        let mut cur = start;
        let result;

        loop {
            if cur < p {
                result = Some(cur);
                break;
            }

            if state[cur] == 2 {
                result = resolved[cur];
                break;
            }

            if state[cur] == 1 {
                result = None;
                break;
            }

            state[cur] = 1;
            path.push(cur);

            match targets[cur] {
                Some(next) => cur = next,
                None => {
                    result = None;
                    break;
                }
            }
        }

        for node in path {
            state[node] = 2;
            resolved[node] = result;
            if result.is_none() {
                invalid_alias[node] = true;
            }
        }
    }

    let mut invalid_alias_names = Vec::new();
    for &id in &alias_ids {
        if invalid_alias[id] {
            invalid_alias_names.push(names[id].clone());
        }
    }
    invalid_alias_names.sort();

    let mut canonicalize = |s: &str| -> Option<usize> {
        let id = ids.get(s).copied()?;
        resolved[id]
    };

    let mut invalid_rules = Vec::new();
    let mut valid_rule_edges = Vec::new();

    for (before, after) in &rules {
        match (canonicalize(before), canonicalize(after)) {
            (Some(x), Some(y)) => valid_rule_edges.push((x, y)),
            _ => invalid_rules.push((before.clone(), after.clone())),
        }
    }

    invalid_rules.sort_by(|x, y| {
        let c = x.0.cmp(&y.0);
        if c == std::cmp::Ordering::Equal {
            x.1.cmp(&y.1)
        } else {
            c
        }
    });

    let mut source_protocol = Vec::with_capacity(k);
    let mut total = vec![0i64; p];
    let mut declarations = vec![0usize; p];
    let mut declaration_seen = vec![false; p];
    let mut conflict = vec![false; p];

    for (idx, (key, &count)) in source_key.iter().zip(source_count.iter()).enumerate() {
        if let Some(c) = canonicalize(key) {
            source_protocol.push(Some(c));
            if declaration_seen[c] {
                conflict[c] = true;
            } else {
                declaration_seen[c] = true;
                declarations[c] = idx;
            }
            total[c] += count;
        } else {
            source_protocol.push(None);
        }
    }

    let mut conflict_names = Vec::new();
    for i in 0..p {
        if conflict[i] {
            conflict_names.push(names[i].clone());
        }
    }
    conflict_names.sort();

    let mut active = vec![false; p];
    let mut active_count = 0usize;
    let mut q: i64 = 0;

    for i in 0..p {
        if total[i] > 0 {
            active[i] = true;
            active_count += 1;
            q += total[i];
        }
    }

    let mut adj = vec![Vec::<usize>::new(); p];
    let mut edge_set = std::collections::HashSet::<(usize, usize)>::with_capacity(valid_rule_edges.len());

    for (x, y) in valid_rule_edges {
        if active[x] && active[y] && edge_set.insert((x, y)) {
            adj[x].push(y);
        }
    }

    let mut indegree = vec![0usize; p];
    for x in 0..p {
        for &y in &adj[x] {
            indegree[y] += 1;
        }
    }

    let mut dfs_state = vec![0u8; p];
    let mut stack_pos = vec![usize::MAX; p];
    let mut dfs_stack: Vec<(usize, usize)> = Vec::new();
    let mut cycle = vec![false; p];

    for start in 0..p {
        if !active[start] || dfs_state[start] != 0 {
            continue;
        }

        dfs_state[start] = 1;
        stack_pos[start] = dfs_stack.len();
        dfs_stack.push((start, 0));

        while let Some((v, ei)) = dfs_stack.last_mut() {
            if *ei == adj[*v].len() {
                let node = *v;
                dfs_state[node] = 2;
                stack_pos[node] = usize::MAX;
                dfs_stack.pop();
                continue;
            }

            let to = adj[*v][*ei];
            *ei += 1;

            if dfs_state[to] == 0 {
                dfs_state[to] = 1;
                stack_pos[to] = dfs_stack.len();
                dfs_stack.push((to, 0));
            } else if dfs_state[to] == 1 {
                let begin = stack_pos[to];
                for j in begin..dfs_stack.len() {
                    cycle[dfs_stack[j].0] = true;
                }
            }
        }
    }

    let mut cycle_names = Vec::new();
    for i in 0..p {
        if cycle[i] {
            cycle_names.push(names[i].clone());
        }
    }
    cycle_names.sort();

    let mut order_error = false;
    let mut order = Vec::new();

    if cycle_names.is_empty() {
        let mut indeg = indegree.clone();

        if mode == b'O' {
            let mut ready = BTreeSet::<(usize, usize)>::new();
            for i in 0..p {
                if active[i] && indeg[i] == 0 {
                    ready.insert((declarations[i], i));
                }
            }

            while let Some(&(decl, v)) = ready.iter().next() {
                ready.remove(&(decl, v));
                order.push(v);

                for &to in &adj[v] {
                    indeg[to] -= 1;
                    if indeg[to] == 0 {
                        ready.insert((declarations[to], to));
                    }
                }
            }
        } else {
            let mut ready = BTreeSet::<usize>::new();
            for i in 0..p {
                if active[i] && indeg[i] == 0 {
                    ready.insert(i);
                }
            }

            while !ready.is_empty() {
                if ready.len() != 1 {
                    order_error = true;
                    break;
                }

                let v = *ready.iter().next().unwrap();
                ready.remove(&v);
                order.push(v);

                for &to in &adj[v] {
                    indeg[to] -= 1;
                    if indeg[to] == 0 {
                        ready.insert(to);
                    }
                }
            }
        }

        if order.len() != active_count {
            order_error = true;
        }
    }

    let error_count =
        invalid_alias_names.len()
        + invalid_rules.len()
        + conflict_names.len()
        + cycle_names.len()
        + if order_error { 1 } else { 0 };

    let mut out = io::BufWriter::new(io::stdout().lock());

    if error_count > 0 {
        writeln!(out, "INVALID {}", error_count).unwrap();

        for name in invalid_alias_names {
            writeln!(out, "ALIAS {}", name).unwrap();
        }

        for (before, after) in invalid_rules {
            writeln!(out, "RULE {} {}", before, after).unwrap();
        }

        for name in conflict_names {
            writeln!(out, "CONFLICT {}", name).unwrap();
        }

        for name in cycle_names {
            writeln!(out, "CYCLE {}", name).unwrap();
        }

        if order_error {
            writeln!(out, "ORDER").unwrap();
        }
    } else {
        writeln!(out, "VALID {}", q).unwrap();

        for protocol in order {
            if total[protocol] == 1 {
                writeln!(out, "{} {}", names[protocol], names[protocol]).unwrap();
            } else {
                for j in 1..=total[protocol] {
                    writeln!(
                        out,
                        "{} {}#{}",
                        names[protocol],
                        names[protocol],
                        j
                    ).unwrap();
                }
            }
        }
    }
}