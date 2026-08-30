use std::io::{self, Read, Write};
use std::collections::BTreeSet;

struct Parser {
    toks: Vec<String>,
    pos: usize,
}

enum Node {
    Var(i64),
    Not(Box<Node>),
    And(Vec<Node>),
    Or(Vec<Node>),
}

fn parse(p: &mut Parser) -> Node {
    let t = p.toks[p.pos].clone();
    p.pos += 1;
    if t == "x" {
        let i: i64 = p.toks[p.pos].parse().unwrap();
        p.pos += 1;
        Node::Var(i)
    } else if t == "!" {
        let c = parse(p);
        Node::Not(Box::new(c))
    } else if t == "&" {
        let k: usize = p.toks[p.pos].parse().unwrap();
        p.pos += 1;
        let mut v = Vec::new();
        for _ in 0..k {
            v.push(parse(p));
        }
        Node::And(v)
    } else {
        let k: usize = p.toks[p.pos].parse().unwrap();
        p.pos += 1;
        let mut v = Vec::new();
        for _ in 0..k {
            v.push(parse(p));
        }
        Node::Or(v)
    }
}

fn push_neg(n: &Node, neg: bool) -> Node {
    match n {
        Node::Var(i) => {
            if neg {
                Node::Var(-*i)
            } else {
                Node::Var(*i)
            }
        }
        Node::Not(c) => push_neg(c, !neg),
        Node::And(v) => {
            let ch: Vec<Node> = v.iter().map(|c| push_neg(c, neg)).collect();
            if neg {
                Node::Or(ch)
            } else {
                Node::And(ch)
            }
        }
        Node::Or(v) => {
            let ch: Vec<Node> = v.iter().map(|c| push_neg(c, neg)).collect();
            if neg {
                Node::And(ch)
            } else {
                Node::Or(ch)
            }
        }
    }
}

struct Ctx {
    n: usize,
    closure: Vec<Vec<usize>>,
    limit: usize,
}

fn idx(n: usize, lit: i64) -> usize {
    if lit > 0 {
        (lit as usize) - 1
    } else {
        n + ((-lit) as usize) - 1
    }
}

fn lit_of(n: usize, id: usize) -> i64 {
    if id < n {
        (id + 1) as i64
    } else {
        -(((id - n) + 1) as i64)
    }
}

type Branch = Vec<usize>;

fn saturate(ctx: &Ctx, base: &BTreeSet<usize>) -> Option<Branch> {
    let mut set: BTreeSet<usize> = BTreeSet::new();
    for &b in base.iter() {
        set.insert(b);
        for &c in ctx.closure[b].iter() {
            set.insert(c);
        }
    }
    let n = ctx.n;
    for &b in set.iter() {
        let comp = if b < n { b + n } else { b - n };
        if set.contains(&comp) {
            return None;
        }
    }
    let mut v: Vec<usize> = set.into_iter().collect();
    v.sort_by_key(|&id| {
        let l = lit_of(n, id);
        (l.abs(), if l > 0 { 0 } else { 1 })
    });
    Some(v)
}

fn subsumes(a: &Branch, b: &Branch) -> bool {
    if a.len() > b.len() {
        return false;
    }
    for x in a.iter() {
        if !b.contains(x) {
            return false;
        }
    }
    true
}

fn insert(list: &mut Vec<Branch>, br: Branch, limit: usize) -> bool {
    for e in list.iter() {
        if subsumes(e, &br) {
            return true;
        }
    }
    list.retain(|e| !subsumes(&br, e));
    list.push(br);
    if list.len() > limit {
        return false;
    }
    true
}

fn canon(list: &mut Vec<Branch>, n: usize) {
    list.sort_by(|a, b| {
        let av: Vec<i64> = a.iter().map(|&x| lit_of(n, x)).collect();
        let bv: Vec<i64> = b.iter().map(|&x| lit_of(n, x)).collect();
        av.cmp(&bv)
    });
}

fn eval(ctx: &Ctx, node: &Node) -> Result<Vec<Branch>, ()> {
    match node {
        Node::Var(i) => {
            let mut s = BTreeSet::new();
            s.insert(idx(ctx.n, *i));
            match saturate(ctx, &s) {
                Some(b) => {
                    let mut v: Vec<Branch> = Vec::new();
                    if !insert(&mut v, b, ctx.limit) {
                        return Err(());
                    }
                    Ok(v)
                }
                None => Ok(Vec::new()),
            }
        }
        Node::Or(ch) => {
            let mut cur = eval(ctx, &ch[0])?;
            for c in ch.iter().skip(1) {
                canon(&mut cur, ctx.n);
                let right = eval(ctx, c)?;
                for r in right.into_iter() {
                    if !insert(&mut cur, r, ctx.limit) {
                        return Err(());
                    }
                }
            }
            canon(&mut cur, ctx.n);
            Ok(cur)
        }
        Node::And(ch) => {
            let mut cur = eval(ctx, &ch[0])?;
            for c in ch.iter().skip(1) {
                canon(&mut cur, ctx.n);
                let right = eval(ctx, c)?;
                let mut nw: Vec<Branch> = Vec::new();
                for a in cur.iter() {
                    for b in right.iter() {
                        let mut s: BTreeSet<usize> = BTreeSet::new();
                        for &x in a.iter() {
                            s.insert(x);
                        }
                        for &x in b.iter() {
                            s.insert(x);
                        }
                        if let Some(u) = saturate(ctx, &s) {
                            if !insert(&mut nw, u, ctx.limit) {
                                return Err(());
                            }
                        }
                    }
                }
                cur = nw;
            }
            canon(&mut cur, ctx.n);
            Ok(cur)
        }
        Node::Not(_) => unreachable!(),
    }
}

fn main() {
    let mut inp = String::new();
    io::stdin().read_to_string(&mut inp).unwrap();
    let toks: Vec<String> = inp.split_ascii_whitespace().map(|s| s.to_string()).collect();
    let mut pos = 0usize;
    let n: usize = toks[pos].parse().unwrap();
    pos += 1;
    let l: usize = toks[pos].parse().unwrap();
    pos += 1;
    let r: usize = toks[pos].parse().unwrap();
    pos += 1;

    let m = 2 * n;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for _ in 0..r {
        let a: i64 = toks[pos].parse().unwrap();
        pos += 1;
        let b: i64 = toks[pos].parse().unwrap();
        pos += 1;
        adj[idx(n, a)].push(idx(n, b));
        adj[idx(n, -b)].push(idx(n, -a));
    }

    let mut closure: Vec<Vec<usize>> = vec![Vec::new(); m];
    for s in 0..m {
        let mut seen = vec![false; m];
        let mut stack = vec![s];
        seen[s] = true;
        let mut out = Vec::new();
        while let Some(u) = stack.pop() {
            for &v in adj[u].iter() {
                if !seen[v] {
                    seen[v] = true;
                    out.push(v);
                    stack.push(v);
                }
            }
        }
        closure[s] = out;
    }

    let mut p = Parser {
        toks: toks[pos..].to_vec(),
        pos: 0,
    };
    let root = parse(&mut p);
    let root = push_neg(&root, false);

    let ctx = Ctx {
        n,
        closure,
        limit: l,
    };

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    match eval(&ctx, &root) {
        Err(_) => {
            writeln!(out, "LIMIT").unwrap();
        }
        Ok(mut res) => {
            canon(&mut res, n);
            writeln!(out, "{}", res.len()).unwrap();
            for b in res.iter() {
                let mut s = String::new();
                s.push_str(&format!("{}", b.len()));
                for &x in b.iter() {
                    s.push(' ');
                    s.push_str(&format!("{}", lit_of(n, x)));
                }
                writeln!(out, "{}", s).unwrap();
            }
        }
    }
}