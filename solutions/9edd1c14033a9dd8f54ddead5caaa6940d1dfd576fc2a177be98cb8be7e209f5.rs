use std::collections::HashMap;
use std::io::{self, Read, Write};

#[derive(Clone, Copy)]
struct PathNode {
    parent: u32,
    start: u32,
    len: u32,
}

struct TrieNode {
    next: [i32; 37],
    category: i32,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            next: [-1; 37],
            category: -1,
        }
    }
}

fn trie_index(b: u8) -> Option<usize> {
    match b {
        b'a'..=b'z' => Some((b - b'a') as usize),
        b'0'..=b'9' => Some(26 + (b - b'0') as usize),
        b'.' => Some(36),
        _ => None,
    }
}

fn hex_value(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

fn lower_ascii(b: u8) -> u8 {
    if b'A' <= b && b <= b'Z' {
        b + (b'a' - b'A')
    } else {
        b
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_ascii_whitespace();

    let c: usize = it.next().unwrap().parse().unwrap();
    let r: usize = it.next().unwrap().parse().unwrap();
    let n: usize = it.next().unwrap().parse().unwrap();

    let mut names: Vec<&str> = Vec::with_capacity(c);
    let mut media: Vec<&str> = Vec::with_capacity(c);
    let mut kinds: Vec<u8> = Vec::with_capacity(c);
    let mut category_map: HashMap<&str, usize> = HashMap::with_capacity(c * 2);

    for id in 0..c {
        let name = it.next().unwrap();
        let med = it.next().unwrap();
        let kind = it.next().unwrap().as_bytes()[0];
        names.push(name);
        media.push(med);
        kinds.push(kind);
        category_map.insert(name, id);
    }

    let mut trie = Vec::with_capacity(300_001);
    trie.push(TrieNode::new());

    for _ in 0..r {
        let suffix = it.next().unwrap().as_bytes();
        let cat_name = it.next().unwrap();
        let cat = *category_map.get(cat_name).unwrap() as i32;

        let mut node = 0usize;
        for &b in suffix.iter().rev() {
            let idx = trie_index(b).unwrap();
            let next = trie[node].next[idx];
            if next == -1 {
                let ni = trie.len();
                trie.push(TrieNode::new());
                trie[node].next[idx] = ni as i32;
                node = ni;
            } else {
                node = next as usize;
            }
        }
        trie[node].category = cat;
    }

    let mut path_nodes: Vec<PathNode> = Vec::with_capacity(2_000_001);
    path_nodes.push(PathNode {
        parent: 0,
        start: 0,
        len: 0,
    });

    let mut arena: Vec<u8> = Vec::with_capacity(2_000_000);
    let mut request_node = vec![0u32; n + 1];
    let mut request_dir = vec![true; n + 1];

    let mut totals = vec![0u64; c];
    let mut unknown = 0u64;

    for i in 1..=n {
        let weight: u64 = it.next().unwrap().parse().unwrap();
        let base: usize = it.next().unwrap().parse().unwrap();
        let target = it.next().unwrap().as_bytes();

        let mut cut = target.len();
        for j in 0..target.len() {
            if target[j] == b'?' || target[j] == b'#' {
                cut = j;
                break;
            }
        }

        let raw = &target[..cut];
        let mut p = Vec::with_capacity(raw.len());
        let mut j = 0;
        while j < raw.len() {
            if raw[j] == b'%' {
                let v = (hex_value(raw[j + 1]) << 4) | hex_value(raw[j + 2]);
                p.push(v);
                j += 3;
            } else {
                p.push(raw[j]);
                j += 1;
            }
        }

        let absolute = p.first() == Some(&b'/');

        let mut current = if absolute {
            0u32
        } else {
            let bnode = request_node[base];
            if request_dir[base] {
                bnode
            } else {
                path_nodes[bnode as usize].parent
            }
        };

        let syntactic_dir = if p.is_empty() || p.last() == Some(&b'/') {
            true
        } else {
            let mut s = p.len();
            while s > 0 && p[s - 1] != b'/' {
                s -= 1;
            }
            let last = &p[s..];
            last == b"." || last == b".."
        };

        let mut pos = 0usize;
        while pos <= p.len() {
            let mut end = pos;
            while end < p.len() && p[end] != b'/' {
                end += 1;
            }

            if end > pos {
                let comp = &p[pos..end];
                if comp == b".." {
                    if current != 0 {
                        current = path_nodes[current as usize].parent;
                    }
                } else if comp != b"." {
                    let start = arena.len() as u32;
                    arena.extend_from_slice(comp);
                    let node = PathNode {
                        parent: current,
                        start,
                        len: comp.len() as u32,
                    };
                    path_nodes.push(node);
                    current = (path_nodes.len() - 1) as u32;
                }
            }

            if end == p.len() {
                break;
            }
            pos = end + 1;
        }

        request_node[i] = current;
        request_dir[i] = syntactic_dir;

        if !syntactic_dir {
            let pn = path_nodes[current as usize];
            let start = pn.start as usize;
            let len = pn.len as usize;
            let comp = &arena[start..start + len];

            let mut tnode = 0usize;
            let mut matched = -1i32;

            for k in 0..len {
                let b = lower_ascii(comp[len - 1 - k]);
                let idx = match trie_index(b) {
                    Some(x) => x,
                    None => break,
                };

                let next = trie[tnode].next[idx];
                if next == -1 {
                    break;
                }
                tnode = next as usize;

                let suffix_len = k + 1;
                if trie[tnode].category >= 0
                    && len > suffix_len + 1
                    && comp[len - suffix_len - 1] == b'.'
                {
                    matched = trie[tnode].category;
                }
            }

            if matched >= 0 {
                totals[matched as usize] += weight;
            } else {
                unknown += weight;
            }
        }
    }

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for i in 0..c {
        if kinds[i] == b'T' {
            writeln!(
                out,
                "{} {} {}{}",
                names[i],
                totals[i],
                media[i],
                ";charset=utf-8"
            )
            .unwrap();
        } else {
            writeln!(out, "{} {} {}", names[i], totals[i], media[i]).unwrap();
        }
    }
    writeln!(out, "UNKNOWN {}", unknown).unwrap();
}