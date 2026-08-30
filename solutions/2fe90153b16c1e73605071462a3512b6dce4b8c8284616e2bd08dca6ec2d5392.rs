use std::io::{Read, Write};

#[derive(Clone, Copy)]
enum F {
    Ex(usize),
    Q {
        kind: u8,
        cc: char,
        op: char,
        cl: char,
        dep: usize,
        interp: bool,
        here: bool,
    },
}

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let mut pos = 0usize;
    while pos < buf.len() && buf[pos] != b'\n' {
        pos += 1;
    }
    let head = String::from_utf8_lossy(&buf[..pos]).trim().to_string();
    let n: usize = head.parse().unwrap_or(0);
    let start = if pos < buf.len() { pos + 1 } else { buf.len() };
    let end = std::cmp::min(buf.len(), start + n);
    let src = String::from_utf8_lossy(&buf[start..end]).to_string();
    let ch: Vec<char> = src.chars().collect();
    let n = ch.len();
    let mut out = vec![b'E'; n];
    let mut lead = vec![false; n + 1];
    if n > 0 {
        lead[0] = true;
    }
    for i in 1..n {
        lead[i] = if ch[i - 1] == '\n' {
            true
        } else {
            lead[i - 1] && (ch[i - 1] == ' ' || ch[i - 1] == '\t')
        };
    }
    let is_here = |p: usize| -> bool {
        let mut j = p;
        while j < n && (ch[j] == ' ' || ch[j] == '\t') {
            j += 1;
        }
        j < n && ch[j] == '\n'
    };
    let mut st: Vec<F> = vec![F::Ex(0)];
    let mut i = 0usize;
    while i < n {
        let l = st.len() - 1;
        let top = st[l];
        match top {
            F::Ex(_) => {
                let c = ch[i];
                if c == '#' {
                    let mut j = i;
                    while j < n && ch[j] != '\n' {
                        out[j] = b'C';
                        j += 1;
                    }
                    i = j;
                } else if c == '/' && i + 1 < n && ch[i + 1] == '*' {
                    let mut j = i;
                    let mut dep = 0usize;
                    while j < n {
                        if j + 1 < n && ch[j] == '/' && ch[j + 1] == '*' {
                            dep += 1;
                            out[j] = b'C';
                            out[j + 1] = b'C';
                            j += 2;
                        } else if j + 1 < n && ch[j] == '*' && ch[j + 1] == '/' {
                            out[j] = b'C';
                            out[j + 1] = b'C';
                            j += 2;
                            if dep > 0 {
                                dep -= 1;
                            }
                            if dep == 0 {
                                break;
                            }
                        } else {
                            out[j] = b'C';
                            j += 1;
                        }
                    }
                    i = j;
                } else if c == '"' || c == '\'' {
                    let trip = i + 2 < n && ch[i + 1] == c && ch[i + 2] == c;
                    if trip {
                        out[i] = b'Q';
                        out[i + 1] = b'Q';
                        out[i + 2] = b'Q';
                        let after = i + 3;
                        let h = is_here(after);
                        st.push(F::Q {
                            kind: 0,
                            cc: c,
                            op: c,
                            cl: c,
                            dep: 1,
                            interp: c == '"',
                            here: h,
                        });
                        i = after;
                    } else {
                        out[i] = b'Q';
                        st.push(F::Q {
                            kind: 1,
                            cc: c,
                            op: c,
                            cl: c,
                            dep: 1,
                            interp: c == '"',
                            here: false,
                        });
                        i += 1;
                    }
                } else if c == '~'
                    && i + 2 < n
                    && ch[i + 1].is_ascii_alphabetic()
                    && {
                        let d = ch[i + 2];
                        d == '('
                            || d == '['
                            || d == '{'
                            || d == '<'
                            || d == '/'
                            || d == '|'
                            || ((d == '"' || d == '\'')
                                && i + 4 < n
                                && ch[i + 3] == d
                                && ch[i + 4] == d)
                    }
                {
                    let d = ch[i + 2];
                    let interp = ch[i + 1].is_ascii_lowercase();
                    if d == '(' || d == '[' || d == '{' || d == '<' {
                        let clo = match d {
                            '(' => ')',
                            '[' => ']',
                            '{' => '}',
                            _ => '>',
                        };
                        out[i] = b'Q';
                        out[i + 1] = b'Q';
                        out[i + 2] = b'Q';
                        st.push(F::Q {
                            kind: 2,
                            cc: d,
                            op: d,
                            cl: clo,
                            dep: 1,
                            interp,
                            here: false,
                        });
                        i += 3;
                    } else if d == '/' || d == '|' {
                        out[i] = b'Q';
                        out[i + 1] = b'Q';
                        out[i + 2] = b'Q';
                        st.push(F::Q {
                            kind: 1,
                            cc: d,
                            op: d,
                            cl: d,
                            dep: 1,
                            interp,
                            here: false,
                        });
                        i += 3;
                    } else {
                        for k in i..i + 5 {
                            out[k] = b'Q';
                        }
                        let after = i + 5;
                        let h = is_here(after);
                        st.push(F::Q {
                            kind: 0,
                            cc: d,
                            op: d,
                            cl: d,
                            dep: 1,
                            interp,
                            here: h,
                        });
                        i = after;
                    }
                } else if c == '?' {
                    out[i] = b'E';
                    if i + 1 < n {
                        out[i + 1] = b'E';
                        if ch[i + 1] == '\\' && i + 2 < n {
                            out[i + 2] = b'E';
                            i += 3;
                        } else {
                            i += 2;
                        }
                    } else {
                        i += 1;
                    }
                } else if c == '{' {
                    if let F::Ex(d) = &mut st[l] {
                        *d += 1;
                    }
                    i += 1;
                } else if c == '}' {
                    let mut popit = false;
                    if let F::Ex(d) = &mut st[l] {
                        if *d > 0 {
                            *d -= 1;
                            if *d == 0 {
                                popit = true;
                            }
                        }
                    }
                    i += 1;
                    if popit && st.len() > 1 {
                        st.pop();
                    }
                } else {
                    i += 1;
                }
            }
            F::Q {
                kind,
                cc,
                op,
                cl,
                dep,
                interp,
                here,
            } => {
                let c = ch[i];
                if c == '\\' {
                    out[i] = b'Q';
                    if i + 1 < n {
                        out[i + 1] = b'Q';
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else if interp && c == '#' && i + 1 < n && ch[i + 1] == '{' {
                    out[i] = b'E';
                    out[i + 1] = b'E';
                    st.push(F::Ex(1));
                    i += 2;
                } else if kind == 0
                    && c == cc
                    && i + 2 < n
                    && ch[i + 1] == cc
                    && ch[i + 2] == cc
                    && (!here || lead[i])
                {
                    out[i] = b'Q';
                    out[i + 1] = b'Q';
                    out[i + 2] = b'Q';
                    st.pop();
                    i += 3;
                } else if kind == 1 && c == cc {
                    out[i] = b'Q';
                    st.pop();
                    i += 1;
                } else if kind == 2 {
                    out[i] = b'Q';
                    if c == op {
                        if let F::Q { dep: d, .. } = &mut st[l] {
                            *d += 1;
                        }
                    } else if c == cl {
                        let mut popit = false;
                        if let F::Q { dep: d, .. } = &mut st[l] {
                            if *d > 0 {
                                *d -= 1;
                            }
                            if *d == 0 {
                                popit = true;
                            }
                        }
                        if popit {
                            st.pop();
                        }
                    }
                    let _ = dep;
                    i += 1;
                } else {
                    out[i] = b'Q';
                    i += 1;
                }
            }
        }
    }
    let so = std::io::stdout();
    let mut w = std::io::BufWriter::new(so.lock());
    w.write_all(&out).unwrap();
    w.write_all(b"\n").unwrap();
}