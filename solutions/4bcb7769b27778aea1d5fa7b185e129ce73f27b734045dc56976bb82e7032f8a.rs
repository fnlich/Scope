use std::io::{self, Read, Write};

fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

fn decode_hex(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut v = Vec::with_capacity(b.len() / 2);
    let mut i = 0;
    while i < b.len() {
        v.push((hex_val(b[i]) << 4) | hex_val(b[i + 1]));
        i += 2;
    }
    v
}

fn hex_string(s: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len() * 2);
    for &b in s {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 15) as usize] as char);
    }
    out
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let mode = it.next().unwrap();
    let pattern = decode_hex(it.next().unwrap());
    let s_byte: u8 = it.next().unwrap().parse().unwrap();
    let r_byte: u8 = it.next().unwrap().parse().unwrap();
    let c: i64 = it.next().unwrap().parse().unwrap();
    let b: i64 = it.next().unwrap().parse().unwrap();
    let q: usize = it.next().unwrap().parse().unwrap();

    let m = pattern.len();
    let mut pi = vec![0usize; m];
    {
        let mut j = 0usize;
        for i in 1..m {
            while j > 0 && pattern[i] != pattern[j] {
                j = pi[j - 1];
            }
            if pattern[i] == pattern[j] {
                j += 1;
            }
            pi[i] = j;
        }
    }

    let max_cap = c + b;
    let mut cap = c;
    let mut data: Vec<u8> = Vec::new();
    let mut head = 0usize;
    let mut retained_start: i64 = 0;
    let mut kmp = 0usize;
    let mut latest_end: Option<usize> = None;

    let mut batches: Vec<(u8, i64, usize, usize)> = Vec::new();

    let mut status: u8 = b'E';
    let mut status_pos: i64 = -1;
    let mut first_s: i64 = -1;
    let mut pos: i64 = 0;

    let mut emit_latest = || {
        if let Some(end_len) = latest_end {
            let n = end_len;
            let start = head;
            batches.push((b'L', retained_start, start, n));
            head += n;
            retained_start += n as i64;
            latest_end = None;
        }
    };

    'chunks: for _ in 0..q {
        let tok = it.next().unwrap();
        if tok == "-" {
            emit_latest();
            continue;
        }

        let bytes = tok.as_bytes();
        let mut i = 0usize;

        while i < bytes.len() {
            let original_pos = pos;
            let raw = (hex_val(bytes[i]) << 4) | hex_val(bytes[i + 1]);
            i += 2;
            pos += 1;

            if raw == s_byte {
                if first_s == -1 {
                    first_s = original_pos;
                }
                if mode == "STOP" {
                    status = b'S';
                    status_pos = original_pos;
                    break 'chunks;
                }
            }

            let transformed = if mode == "REPLACE" && raw == s_byte {
                r_byte
            } else {
                raw
            };

            let mut retained_len = data.len() - head;

            while retained_len as i64 == cap {
                if latest_end.is_some() {
                    emit_latest();
                    retained_len = data.len() - head;
                } else {
                    let new_cap = if cap == 0 {
                        1
                    } else if cap <= max_cap / 2 {
                        cap * 2
                    } else {
                        max_cap
                    };

                    if new_cap <= cap {
                        status = b'L';
                        status_pos = original_pos;
                        break 'chunks;
                    }

                    cap = new_cap;
                }
            }

            data.push(transformed);
            retained_len += 1;

            while kmp > 0 && pattern[kmp] != transformed {
                kmp = pi[kmp - 1];
            }
            if pattern[kmp] == transformed {
                kmp += 1;
            }

            if kmp == m {
                latest_end = Some(retained_len);
                kmp = pi[m - 1];
            }
        }

        if status != b'E' {
            break;
        }

        emit_latest();
    }

    if status != b'L' {
        emit_latest();

        if head < data.len() {
            batches.push((
                b'F',
                retained_start,
                head,
                data.len() - head,
            ));
            retained_start += (data.len() - head) as i64;
            head = data.len();
        }
    }

    let mut out = io::BufWriter::new(io::stdout());

    match status {
        b'E' => writeln!(out, "EOF").unwrap(),
        b'S' => writeln!(out, "STOP {}", status_pos).unwrap(),
        b'L' => writeln!(out, "LIMIT {}", status_pos).unwrap(),
        _ => unreachable!(),
    }

    writeln!(out, "FIRST {}", first_s).unwrap();
    writeln!(out, "BATCHES {}", batches.len()).unwrap();

    for &(kind, orig, start, len) in &batches {
        writeln!(
            out,
            "{} {} {} {}",
            kind as char,
            orig,
            len,
            hex_string(&data[start..start + len])
        )
        .unwrap();
    }

    let retained_len = data.len() - head;
    if retained_len == 0 {
        writeln!(out, "R {} 0 -", retained_start).unwrap();
    } else {
        writeln!(
            out,
            "R {} {} {}",
            retained_start,
            retained_len,
            hex_string(&data[head..])
        )
        .unwrap();
    }
}