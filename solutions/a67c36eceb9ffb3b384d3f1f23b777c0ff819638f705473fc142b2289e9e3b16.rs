use std::io::{self, Read, Write};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let n: usize = it.next().unwrap().parse().unwrap();
    let e: usize = it.next().unwrap().parse().unwrap();

    let mut parent = vec![usize::MAX; n];
    for i in 0..n {
        let p: usize = it.next().unwrap().parse().unwrap();
        if p == 0 {
            parent[i] = usize::MAX;
        } else {
            parent[i] = p - 1;
        }
    }

    let mut subtree_end: Vec<usize> = (0..n).collect();
    for i in (0..n).rev() {
        if parent[i] != usize::MAX {
            let p = parent[i];
            if subtree_end[i] > subtree_end[p] {
                subtree_end[p] = subtree_end[i];
            }
        }
    }

    let mut status = vec![0u8; n];
    let mut open = vec![false; n];
    let mut begin_time = vec![0i64; n];
    let mut close_time = vec![0i64; n];
    let mut current: Vec<Option<String>> = (0..n).map(|_| None).collect();
    let mut after: Vec<Option<String>> = (0..n).map(|_| None).collect();
    let mut error: Vec<Option<String>> = (0..n).map(|_| None).collect();

    let mut first_child = vec![usize::MAX; n];
    let mut last_child = vec![usize::MAX; n];
    let mut next_sibling = vec![usize::MAX; n];

    for _ in 0..e {
        let time: i64 = it.next().unwrap().parse().unwrap();
        let kind = it.next().unwrap();

        match kind {
            "BEGIN" => {
                let id: usize = it.next().unwrap().parse::<usize>().unwrap() - 1;
                begin_time[id] = time;
                open[id] = true;
                status[id] = 7;

                let p = parent[id];
                if p != usize::MAX {
                    if first_child[p] == usize::MAX {
                        first_child[p] = id;
                    } else {
                        next_sibling[last_child[p]] = id;
                    }
                    last_child[p] = id;
                }
            }
            "CURRENT" => {
                let id: usize = it.next().unwrap().parse::<usize>().unwrap() - 1;
                let s = it.next().unwrap().to_string();
                current[id] = Some(s);
            }
            "OUTCOME" => {
                let id: usize = it.next().unwrap().parse::<usize>().unwrap() - 1;
                let outcome = it.next().unwrap();
                match outcome {
                    "U" => {
                        status[id] = 1;
                    }
                    "N" => {
                        status[id] = 2;
                    }
                    "F" => {
                        let err = it.next().unwrap().to_string();
                        status[id] = 3;
                        error[id] = Some(err);
                    }
                    _ => {}
                }
            }
            "AFTER" => {
                let id: usize = it.next().unwrap().parse::<usize>().unwrap() - 1;
                let s = it.next().unwrap().to_string();
                after[id] = Some(s);
            }
            "END" => {
                let id: usize = it.next().unwrap().parse::<usize>().unwrap() - 1;
                close_time[id] = time;
                open[id] = false;
            }
            "SKIP" => {
                let id: usize = it.next().unwrap().parse::<usize>().unwrap() - 1;
                let typ = it.next().unwrap();
                let s = if typ == "E" { 5 } else { 6 };
                let end = subtree_end[id];
                for j in id..=end {
                    status[j] = s;
                    open[j] = false;
                }
            }
            "ABORT" => {
                let abort_error = it.next().unwrap().to_string();

                for id in 0..n {
                    if open[id] {
                        close_time[id] = time;
                        open[id] = false;
                        if status[id] == 3 {
                        } else {
                            status[id] = 4;
                            error[id] = Some(abort_error.clone());
                        }
                    } else if status[id] == 0 {
                        error[id] = Some(abort_error.clone());
                    }
                }
            }
            _ => {}
        }
    }

    let mut exclusive = vec![0i64; n];
    let mut inclusive = vec![0i64; n];

    for id in 0..n {
        if status[id] >= 1 && status[id] <= 4 {
            inclusive[id] = close_time[id] - begin_time[id];
        }
    }

    for p in 0..n {
        if status[p] < 1 || status[p] > 4 {
            continue;
        }

        let mut covered = 0i64;
        let mut have = false;
        let mut union_start = 0i64;
        let mut union_end = 0i64;

        let mut child = first_child[p];
        while child != usize::MAX {
            if status[child] >= 1 && status[child] <= 4 {
                let s = begin_time[child];
                let t = close_time[child];

                if !have {
                    union_start = s;
                    union_end = t;
                    have = true;
                } else if s <= union_end {
                    if t > union_end {
                        union_end = t;
                    }
                } else {
                    covered += union_end - union_start;
                    union_start = s;
                    union_end = t;
                }
            }
            child = next_sibling[child];
        }

        if have {
            covered += union_end - union_start;
        }

        exclusive[p] = inclusive[p] - covered;
    }

    let mut updated = 0usize;
    let mut unchanged = 0usize;
    let mut failed = 0usize;
    let mut interrupted = 0usize;
    let mut skip_e = 0usize;
    let mut skip_t = 0usize;
    let mut unprocessed = 0usize;

    for &s in &status {
        match s {
            1 => updated += 1,
            2 => unchanged += 1,
            3 => failed += 1,
            4 => interrupted += 1,
            5 => skip_e += 1,
            6 => skip_t += 1,
            _ => unprocessed += 1,
        }
    }

    let mut out = io::BufWriter::new(io::stdout().lock());

    writeln!(
        out,
        "{} {} {} {} {} {} {} {}",
        n, updated, unchanged, failed, interrupted, skip_e, skip_t, unprocessed
    )
    .unwrap();

    for id in 0..n {
        let status_name = match status[id] {
            1 => "UPDATED",
            2 => "UNCHANGED",
            3 => "FAILED",
            4 => "INTERRUPTED",
            5 => "SKIP_E",
            6 => "SKIP_T",
            _ => "UNPROCESSED",
        };

        let current_text = current[id].as_deref().unwrap_or("-");
        let after_text = after[id].as_deref().unwrap_or("-");
        let error_text = error[id].as_deref().unwrap_or("-");

        writeln!(
            out,
            "{} {} {} {} {} {} {}",
            id + 1,
            status_name,
            current_text,
            after_text,
            error_text,
            inclusive[id],
            exclusive[id]
        )
        .unwrap();
    }
}