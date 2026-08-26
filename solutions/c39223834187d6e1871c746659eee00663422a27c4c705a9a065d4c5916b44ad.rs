use std::io::{self, Read, Write};

fn read_i64(data: &[u8], pos: &mut usize) -> i64 {
    while *pos < data.len() && data[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
    let mut sign = 1i64;
    if *pos < data.len() && data[*pos] == b'-' {
        sign = -1;
        *pos += 1;
    }
    let mut value = 0i64;
    while *pos < data.len() && data[*pos].is_ascii_digit() {
        value = value * 10 + (data[*pos] - b'0') as i64;
        *pos += 1;
    }
    value * sign
}

fn main() {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input).unwrap();

    let mut pos = 0usize;
    let l = read_i64(&input, &mut pos) as usize;
    let s = read_i64(&input, &mut pos) as usize;

    let mut w = [0i64; 676];
    for i in 0..676 {
        w[i] = read_i64(&input, &mut pos);
    }

    let mut line_end = pos;
    while line_end < input.len() && input[line_end] != b'\n' {
        line_end += 1;
    }
    let cipher_start = if line_end < input.len() {
        line_end + 1
    } else {
        line_end
    };

    let ciphertext = &input[cipher_start..];

    let mut c = Vec::with_capacity(ciphertext.len());
    for &b in ciphertext {
        if b'A' <= b && b <= b'Z' {
            c.push(b - b'A');
        } else if b'a' <= b && b <= b'z' {
            c.push(b - b'a');
        }
    }

    let mut out = io::BufWriter::new(io::stdout());

    if c.is_empty() {
        out.write_all(b"NO-PLAINTEXT").unwrap();
        return;
    }

    let mut transformed = vec![0i64; 676 * 676];
    for x in 0..26 {
        for y in 0..26 {
            let base = (x * 26 + y) * 676;
            for a in 0..26 {
                let px = if x >= a { x - a } else { x + 26 - a };
                for b in 0..26 {
                    let py = if y >= b { y - b } else { y + 26 - b };
                    transformed[base + a * 26 + b] = w[px * 26 + py];
                }
            }
        }
    }

    let mut edge = vec![0i64; l * 676];
    for t in 0..c.len() - 1 {
        let r = t % l;
        let pair = c[t] as usize * 26 + c[t + 1] as usize;
        let src = pair * 676;
        let dst = r * 676;
        for j in 0..676 {
            edge[dst + j] += transformed[src + j];
        }
    }

    let neg = -4_000_000_000_000_000_000i64;

    let mut next = vec![neg; 676];
    let mut cur = vec![neg; 676];

    let mut tail_index = [0usize; 676];
    for rem in 0..26 {
        for val in 0..26 {
            let nr = if rem >= val {
                rem - val
            } else {
                rem + 26 - val
            };
            tail_index[rem * 26 + val] = nr * 26 + val;
        }
    }

    let mut best_score = neg;
    let mut best_start = 0usize;

    for start in 0..26 {
        next.fill(neg);

        for prev in 0..26 {
            next[prev] = edge[(l - 1) * 676 + prev * 26 + start];
        }

        for r in (1..l).rev() {
            cur.fill(neg);
            let edge_base = (r - 1) * 676;

            for rem in 0..26 {
                let state_base = rem * 26;
                for prev in 0..26 {
                    let mut best = neg;
                    let eb = edge_base + prev * 26;

                    for val in 0..26 {
                        let tail = next[tail_index[state_base + val]];
                        if tail != neg {
                            let candidate = edge[eb + val] + tail;
                            if candidate > best {
                                best = candidate;
                            }
                        }
                    }

                    cur[state_base + prev] = best;
                }
            }

            std::mem::swap(&mut cur, &mut next);
        }

        let required = if s >= start {
            s - start
        } else {
            s + 26 - start
        };

        let score = next[required * 26 + start];

        if score > best_score || (score == best_score && start < best_start) {
            best_score = score;
            best_start = start;
        }
    }

    let start = best_start;
    let required = if s >= start {
        s - start
    } else {
        s + 26 - start
    };

    let mut suffix = vec![neg; (l + 1) * 676];
    let base_l = l * 676;

    for prev in 0..26 {
        suffix[base_l + prev] = edge[(l - 1) * 676 + prev * 26 + start];
    }

    for r in (1..l).rev() {
        let next_base = (r + 1) * 676;
        let cur_base = r * 676;
        let edge_base = (r - 1) * 676;

        for rem in 0..26 {
            let state_base = rem * 26;

            for prev in 0..26 {
                let mut best = neg;
                let eb = edge_base + prev * 26;

                for val in 0..26 {
                    let tail = suffix[next_base + tail_index[state_base + val]];
                    if tail != neg {
                        let candidate = edge[eb + val] + tail;
                        if candidate > best {
                            best = candidate;
                        }
                    }
                }

                suffix[cur_base + state_base + prev] = best;
            }
        }
    }

    let mut key = vec![0u8; l];
    key[0] = start as u8;

    let mut prev = start;
    let mut rem = required;

    for r in 1..l {
        let target = suffix[r * 676 + rem * 26 + prev];
        let edge_base = (r - 1) * 676 + prev * 26;
        let next_base = (r + 1) * 676;

        for val in 0..26 {
            let new_rem = if rem >= val {
                rem - val
            } else {
                rem + 26 - val
            };

            let tail = suffix[next_base + new_rem * 26 + val];
            let candidate = edge[edge_base + val] + tail;

            if candidate == target {
                key[r] = val as u8;
                prev = val;
                rem = new_rem;
                break;
            }
        }
    }

    write!(out, "{}\n", best_score).unwrap();

    for &v in &key {
        out.write_all(&[b'A' + v]).unwrap();
    }
    out.write_all(b"\n").unwrap();

    let mut plaintext = ciphertext.to_vec();
    let mut letter_pos = 0usize;

    for b in &mut plaintext {
        if b'A' <= *b && *b <= b'Z' {
            let x = (*b - b'A') as i32;
            let k = key[letter_pos % l] as i32;
            *b = b'A' + ((x - k + 26) % 26) as u8;
            letter_pos += 1;
        } else if b'a' <= *b && *b <= b'z' {
            let x = (*b - b'a') as i32;
            let k = key[letter_pos % l] as i32;
            *b = b'a' + ((x - k + 26) % 26) as u8;
            letter_pos += 1;
        }
    }

    out.write_all(&plaintext).unwrap();
    out.flush().unwrap();
}