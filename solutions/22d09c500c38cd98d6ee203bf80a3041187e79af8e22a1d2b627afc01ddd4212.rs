use std::io::{Read, Write};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum V { Absent, T, F, Ref }

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_ascii_whitespace();
    let d: usize = it.next().unwrap().parse().unwrap();
    let c: usize = it.next().unwrap().parse().unwrap();

    let mut alias_type: HashMap<String, String> = HashMap::new();
    let mut index_type: HashMap<String, String> = HashMap::new();
    let mut idx_ids: HashMap<String, usize> = HashMap::new();
    let mut idx_names: Vec<String> = Vec::new();
    let mut alias_ids: HashMap<String, usize> = HashMap::new();
    let mut alias_names: Vec<String> = Vec::new();
    let mut decls: Vec<(usize, usize)> = Vec::new();
    let mut conflict = false;

    for _ in 0..d {
        let a = it.next().unwrap().to_ascii_lowercase();
        let i = it.next().unwrap().to_ascii_lowercase();
        let t = it.next().unwrap().to_string();
        match alias_type.get(&a) {
            Some(x) => { if *x != t { conflict = true; } }
            None => { alias_type.insert(a.clone(), t.clone()); }
        }
        match index_type.get(&i) {
            Some(x) => { if *x != t { conflict = true; } }
            None => { index_type.insert(i.clone(), t.clone()); }
        }
        let ai = *alias_ids.entry(a.clone()).or_insert_with(|| { alias_names.push(a.clone()); alias_names.len() - 1 });
        let ii = *idx_ids.entry(i.clone()).or_insert_with(|| { idx_names.push(i.clone()); idx_names.len() - 1 });
        decls.push((ai, ii));
    }

    let mut deps: Vec<(usize, usize)> = Vec::new();
    for _ in 0..c {
        let f = it.next().unwrap().to_ascii_lowercase();
        let t = it.next().unwrap().to_ascii_lowercase();
        match (idx_ids.get(&f), idx_ids.get(&t)) {
            (Some(a), Some(b)) => deps.push((*a, *b)),
            _ => { conflict = true; }
        }
    }

    // parse filter iteratively
    #[derive(Clone)]
    enum Node { Const(bool), R(usize, usize), N(usize), A(Vec<usize>), O(Vec<usize>) }
    // tokens for R: alias type value
    let mut nodes: Vec<Node> = Vec::new();
    let mut refs: Vec<(String, String)> = Vec::new();
    // build with explicit stack
    enum Frame { N, A(usize, Vec<usize>), O(usize, Vec<usize>) }
    let mut stack: Vec<Frame> = Vec::new();
    let mut root: usize = usize::MAX;
    loop {
        // read one node start
        let tok = match it.next() { Some(t) => t, None => break };
        let mut done: Option<usize> = match tok {
            "T" => { nodes.push(Node::Const(true)); Some(nodes.len() - 1) }
            "F" => { nodes.push(Node::Const(false)); Some(nodes.len() - 1) }
            "R" => {
                let a = it.next().unwrap().to_ascii_lowercase();
                let t = it.next().unwrap().to_string();
                let v = it.next().unwrap();
                if v == "NULL" {
                    nodes.push(Node::Const(true));
                    let id = nodes.len() - 1;
                    nodes[id] = Node::R(usize::MAX, 0);
                    Some(id)
                } else {
                    refs.push((a, t));
                    nodes.push(Node::R(refs.len() - 1, 1));
                    Some(nodes.len() - 1)
                }
            }
            "N" => { stack.push(Frame::N); None }
            "A" => {
                let k: usize = it.next().unwrap().parse().unwrap();
                if k == 0 { nodes.push(Node::A(Vec::new())); Some(nodes.len() - 1) }
                else { stack.push(Frame::A(k, Vec::new())); None }
            }
            "O" => {
                let k: usize = it.next().unwrap().parse().unwrap();
                if k == 0 { nodes.push(Node::O(Vec::new())); Some(nodes.len() - 1) }
                else { stack.push(Frame::O(k, Vec::new())); None }
            }
            _ => None,
        };
        while let Some(child) = done {
            match stack.pop() {
                None => { root = child; done = None; }
                Some(Frame::N) => { nodes.push(Node::N(child)); done = Some(nodes.len() - 1); }
                Some(Frame::A(k, mut v)) => {
                    v.push(child);
                    if v.len() == k { nodes.push(Node::A(v)); done = Some(nodes.len() - 1); }
                    else { stack.push(Frame::A(k, v)); done = None; }
                }
                Some(Frame::O(k, mut v)) => {
                    v.push(child);
                    if v.len() == k { nodes.push(Node::O(v)); done = Some(nodes.len() - 1); }
                    else { stack.push(Frame::O(k, v)); done = None; }
                }
            }
        }
        if root != usize::MAX { break; }
    }

    let n = nodes.len();
    let mut val: Vec<V> = vec![V::Absent; n];
    let mut live_children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        match &nodes[i] {
            Node::Const(b) => { val[i] = if *b { V::T } else { V::F }; }
            Node::R(_, k) => { val[i] = if *k == 1 { V::Ref } else { V::Absent }; }
            Node::N(ch) => {
                val[i] = match val[*ch] {
                    V::Absent => V::Absent,
                    V::T => V::F,
                    V::F => V::T,
                    V::Ref => V::Ref,
                };
                if val[i] == V::Ref { live_children[i].push(*ch); }
            }
            Node::A(ch) | Node::O(ch) => {
                let is_and = matches!(&nodes[i], Node::A(_));
                let ann = if is_and { V::F } else { V::T };
                let ident = if is_and { V::T } else { V::F };
                let mut kept: Vec<usize> = Vec::new();
                let mut has_ann = false;
                for &x in ch.iter() {
                    if val[x] == V::Absent { continue; }
                    if val[x] == ann { has_ann = true; }
                    kept.push(x);
                }
                if has_ann {
                    val[i] = ann;
                } else {
                    let mut nonconst: Vec<usize> = Vec::new();
                    for &x in kept.iter() {
                        if val[x] == V::Ref { nonconst.push(x); }
                    }
                    if kept.is_empty() {
                        val[i] = ident;
                    } else if nonconst.is_empty() {
                        val[i] = ident;
                    } else {
                        val[i] = V::Ref;
                        live_children[i] = nonconst;
                    }
                }
            }
        }
    }

    let mut live_ref_ids: Vec<usize> = Vec::new();
    if val[root] == V::Ref {
        let mut st = vec![root];
        while let Some(x) = st.pop() {
            match &nodes[x] {
                Node::R(r, k) => { if *k == 1 { live_ref_ids.push(*r); } }
                _ => { for &y in live_children[x].iter() { st.push(y); } }
            }
        }
    }

    let out = std::io::stdout();
    let mut o = std::io::BufWriter::new(out.lock());

    if conflict {
        writeln!(o, "CONFLICT").unwrap();
        return;
    }

    let mut need_alias: Vec<usize> = Vec::new();
    for &r in live_ref_ids.iter() {
        let (a, t) = &refs[r];
        match alias_type.get(a) {
            Some(dt) if dt == t => { need_alias.push(alias_ids[a]); }
            _ => { writeln!(o, "INVALID").unwrap(); return; }
        }
    }
    need_alias.sort();
    need_alias.dedup();

    let m = idx_names.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&x, &y| idx_names[x].cmp(&idx_names[y]));
    let mut pos = vec![0usize; m];
    for (p, &x) in order.iter().enumerate() { pos[x] = p; }

    let mut adj = vec![0u32; m];
    for &(f, t) in deps.iter() { adj[pos[f]] |= 1u32 << pos[t]; }
    let mut closure = vec![0u32; m];
    for i in 0..m {
        let mut seen: u32 = 1u32 << i;
        let mut st = vec![i];
        while let Some(x) = st.pop() {
            let mut b = adj[x];
            while b != 0 {
                let j = b.trailing_zeros() as usize;
                b &= b - 1;
                if seen & (1u32 << j) == 0 { seen |= 1u32 << j; st.push(j); }
            }
        }
        closure[i] = seen;
    }

    let na = alias_names.len();
    let mut supply = vec![0u64; m];
    let mut need_mask: u64 = 0;
    let mut alias_bit = vec![usize::MAX; na];
    let mut nb = 0usize;
    for &a in need_alias.iter() { alias_bit[a] = nb; need_mask |= 1u64 << nb; nb += 1; }
    for &(ai, ii) in decls.iter() {
        if alias_bit[ai] != usize::MAX {
            supply[pos[ii]] |= 1u64 << alias_bit[ai];
        }
    }

    let mut best: Option<u32> = None;
    let mut best_cnt = usize::MAX;
    for mask in 0u32..(1u32 << m) {
        let cnt = mask.count_ones() as usize;
        if cnt > best_cnt { continue; }
        let mut cl = 0u32;
        let mut b = mask;
        while b != 0 {
            let j = b.trailing_zeros() as usize;
            b &= b - 1;
            cl |= closure[j];
        }
        if cl != mask { continue; }
        let mut sup = 0u64;
        let mut b2 = mask;
        while b2 != 0 {
            let j = b2.trailing_zeros() as usize;
            b2 &= b2 - 1;
            sup |= supply[j];
        }
        if sup & need_mask != need_mask { continue; }
        if cnt < best_cnt || best.is_none() {
            best_cnt = cnt;
            best = Some(mask);
        } else if cnt == best_cnt {
            let cur = best.unwrap();
            if lex_smaller(mask, cur) { best = Some(mask); }
        }
    }

    match best {
        None => { writeln!(o, "INVALID").unwrap(); }
        Some(mask) => {
            let mut res: Vec<&str> = Vec::new();
            for p in 0..m {
                if mask & (1u32 << p) != 0 { res.push(&idx_names[order[p]]); }
            }
            let mut line = String::new();
            line.push_str(&res.len().to_string());
            for r in res.iter() { line.push(' '); line.push_str(r); }
            writeln!(o, "{}", line).unwrap();
        }
    }
}

fn lex_smaller(a: u32, b: u32) -> bool {
    let mut x = a;
    let mut y = b;
    loop {
        if x == 0 { return y != 0; }
        if y == 0 { return false; }
        let i = x.trailing_zeros();
        let j = y.trailing_zeros();
        if i != j { return i < j; }
        x &= x - 1;
        y &= y - 1;
    }
}