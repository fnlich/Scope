use std::collections::HashMap;
use std::io::{self, Read, Write};

const NONE: usize = usize::MAX;

fn main() {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input).unwrap();

    let mut pos = 0usize;
    fn next_token(input: &[u8], pos: &mut usize) -> Vec<u8> {
        while *pos < input.len() && input[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        let start = *pos;
        while *pos < input.len() && !input[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        input[start..*pos].to_vec()
    }

    let q_bytes = next_token(&input, &mut pos);
    let q: usize = std::str::from_utf8(&q_bytes).unwrap().parse().unwrap();

    let mut ids: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut names: Vec<Vec<u8>> = Vec::new();

    let mut fs: Vec<Option<(usize, usize)>> = Vec::new();
    let mut manual: Vec<Option<(usize, usize)>> = Vec::new();
    let mut active: Vec<Option<(usize, usize)>> = Vec::new();
    let mut seen: Vec<u32> = Vec::new();
    let mut stamp: u32 = 0;

    fn get_id(
        name: Vec<u8>,
        ids: &mut HashMap<Vec<u8>, usize>,
        names: &mut Vec<Vec<u8>>,
        fs: &mut Vec<Option<(usize, usize)>>,
        manual: &mut Vec<Option<(usize, usize)>>,
        active: &mut Vec<Option<(usize, usize)>>,
        seen: &mut Vec<u32>,
    ) -> usize {
        if let Some(&id) = ids.get(&name) {
            return id;
        }
        let id = names.len();
        ids.insert(name.clone(), id);
        names.push(name);
        fs.push(None);
        manual.push(None);
        active.push(None);
        seen.push(0);
        id
    }

    fn parse_ref(
        tok: Vec<u8>,
        ids: &mut HashMap<Vec<u8>, usize>,
        names: &mut Vec<Vec<u8>>,
        fs: &mut Vec<Option<(usize, usize)>>,
        manual: &mut Vec<Option<(usize, usize)>>,
        active: &mut Vec<Option<(usize, usize)>>,
        seen: &mut Vec<u32>,
    ) -> usize {
        if tok == b"-" {
            NONE
        } else {
            get_id(tok, ids, names, fs, manual, active, seen)
        }
    }

    let mut out = io::BufWriter::new(io::stdout());

    for _ in 0..q {
        let cmd = next_token(&input, &mut pos);

        match cmd.as_slice() {
            b"FILE" => {
                let name = next_token(&input, &mut pos);
                let first = next_token(&input, &mut pos);
                let second = next_token(&input, &mut pos);

                let id = get_id(
                    name, &mut ids, &mut names, &mut fs, &mut manual, &mut active, &mut seen,
                );
                let a = parse_ref(
                    first, &mut ids, &mut names, &mut fs, &mut manual, &mut active, &mut seen,
                );
                let b = parse_ref(
                    second, &mut ids, &mut names, &mut fs, &mut manual, &mut active, &mut seen,
                );
                fs[id] = Some((a, b));
            }
            b"ERASE" => {
                let name = next_token(&input, &mut pos);
                if let Some(&id) = ids.get(&name) {
                    fs[id] = None;
                }
            }
            b"MANUAL" => {
                let name = next_token(&input, &mut pos);
                let first = next_token(&input, &mut pos);
                let second = next_token(&input, &mut pos);

                let id = get_id(
                    name, &mut ids, &mut names, &mut fs, &mut manual, &mut active, &mut seen,
                );
                let a = parse_ref(
                    first, &mut ids, &mut names, &mut fs, &mut manual, &mut active, &mut seen,
                );
                let b = parse_ref(
                    second, &mut ids, &mut names, &mut fs, &mut manual, &mut active, &mut seen,
                );
                let pair = Some((a, b));
                manual[id] = pair;
                active[id] = pair;
            }
            b"RELOAD" => {
                active.clone_from(&fs);
                for i in 0..manual.len() {
                    if manual[i].is_some() {
                        active[i] = manual[i];
                    }
                }
            }
            b"RENDER" => {
                let name = next_token(&input, &mut pos);

                let start = match ids.get(&name) {
                    Some(&id) if active[id].is_some() => id,
                    _ => {
                        writeln!(out, "MISSING_PAGE").unwrap();
                        continue;
                    }
                };

                stamp = stamp.wrapping_add(1);
                if stamp == 0 {
                    seen.fill(0);
                    stamp = 1;
                }

                let mut cur = start;

                loop {
                    if seen[cur] == stamp {
                        writeln!(out, "CYCLE").unwrap();
                        break;
                    }
                    seen[cur] = stamp;

                    let (first, second) = active[cur].unwrap();

                    if first == NONE {
                        writeln!(out, "FOUND").unwrap();
                        break;
                    }

                    if active[first].is_some() {
                        cur = first;
                        continue;
                    }

                    if second == NONE {
                        writeln!(out, "FOUND").unwrap();
                        break;
                    }

                    if active[second].is_some() {
                        cur = second;
                        continue;
                    }

                    write!(out, "MISSING_LAYOUT ").unwrap();
                    out.write_all(&names[second]).unwrap();
                    out.write_all(b"\n").unwrap();
                    break;
                }
            }
            _ => {}
        }
    }
}