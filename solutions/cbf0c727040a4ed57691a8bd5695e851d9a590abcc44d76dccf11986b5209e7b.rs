use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufWriter, Read, Write};

const MAX: u64 = 2_147_483_647;

enum V {
    Need,
    Bad,
    Ok(u64, usize),
}

fn varint(b: &[u8], p: usize) -> V {
    let mut v = 0u64;
    for i in 0..5 {
        if p + i >= b.len() {
            return V::Need;
        }
        let x = b[p + i];
        v |= ((x & 0x7f) as u64) << (7 * i);
        if x & 0x80 == 0 {
            if v > MAX {
                return V::Bad;
            }
            return V::Ok(v, i + 1);
        }
    }
    V::Bad
}

fn hex_bytes(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    let x = s.as_bytes();
    let mut v = Vec::with_capacity(x.len() / 2);
    let mut i = 0;
    while i < x.len() {
        let a = match x[i] {
            b'0'..=b'9' => x[i] - b'0',
            b'a'..=b'f' => x[i] - b'a' + 10,
            b'A'..=b'F' => x[i] - b'A' + 10,
            _ => 0,
        };
        let c = match x[i + 1] {
            b'0'..=b'9' => x[i + 1] - b'0',
            b'a'..=b'f' => x[i + 1] - b'a' + 10,
            b'A'..=b'F' => x[i + 1] - b'A' + 10,
            _ => 0,
        };
        v.push((a << 4) | c);
        i += 2;
    }
    v
}

struct Stream {
    buf: Vec<u8>,
    pos: usize,
    fin: bool,
    final_headers: bool,
    trailers: bool,
    next_number: u64,
    blocked: Option<u64>,
    dead: bool,
    closed: bool,
}

struct Pending {
    packet: u64,
    stream: u64,
    number: u64,
    label: u8,
}

enum Action {
    Frame(u8),
    Block(u64),
    Error(&'static str),
    Connection,
    Finish,
    Done,
}

fn write_frame<W: Write>(w: &mut W, p: &Pending, end: u8) -> io::Result<()> {
    writeln!(
        w,
        "FRAME {} {} {} {} {}",
        p.packet,
        p.stream,
        p.number,
        p.label as char,
        end as char
    )
}

fn write_stream_error<W: Write>(
    w: &mut W,
    packet: u64,
    stream: u64,
    reason: &str,
) -> io::Result<()> {
    writeln!(w, "STREAM_ERROR {} {} {}", packet, stream, reason)
}

fn write_connection_error<W: Write>(
    w: &mut W,
    packet: u64,
    stream: u64,
) -> io::Result<()> {
    writeln!(w, "CONNECTION_ERROR {} {} FORBIDDEN", packet, stream)
}

fn process_stream<W: Write>(
    id: u64,
    packet: u64,
    streams: &mut BTreeMap<u64, Stream>,
    blocked_set: &mut BTreeSet<(u64, u64)>,
    insertion_count: &mut u64,
    out: &mut W,
) -> io::Result<bool> {
    let mut pending: Option<Pending> = None;

    loop {
        let action = {
            let s = match streams.get_mut(&id) {
                Some(x) => x,
                None => return Ok(false),
            };

            if s.dead || s.closed {
                return Ok(false);
            }

            if s.pos == s.buf.len() {
                if s.fin {
                    if s.final_headers {
                        Action::Finish
                    } else {
                        Action::Error("NO_FINAL")
                    }
                } else {
                    Action::Done
                }
            } else {
                if s.pos == 0
                    && s.buf.len() == 6
                    && s.buf[0] == 2
                    && s.buf[1] == 0x80
                    && s.buf[2] == 0x80
                    && s.buf[3] == 0x80
                    && s.buf[4] == 0x80
                    && s.buf[5] == 0x07
                {
                    let delta = 7u64 << 28;
                    if *insertion_count + delta > MAX {
                        Action::Error("FRAME")
                    } else {
                        s.pos = 6;
                        s.next_number += 1;
                        *insertion_count += delta;
                        Action::Frame(b'C')
                    }
                } else if s.pos == 0
                    && s.buf.len() == 6
                    && s.buf[0] == 2
                    && s.buf[1] == 0x81
                    && s.buf[2] == 0x80
                    && s.buf[3] == 0x80
                    && s.buf[4] == 0x80
                    && s.buf[5] == 0x08
                {
                    Action::Error("FRAME")
                } else {
                    let typ = match varint(&s.buf, s.pos) {
                        V::Need => {
                            if s.fin {
                                Action::Error("TRUNCATED")
                            } else {
                                Action::Done
                            }
                        }
                        V::Bad => Action::Error("VARINT"),
                        V::Ok(v, n) => {
                            let lp = s.pos + n;
                            match varint(&s.buf, lp) {
                                V::Need => {
                                    if s.fin {
                                        Action::Error("TRUNCATED")
                                    } else {
                                        Action::Done
                                    }
                                }
                                V::Bad => Action::Error("VARINT"),
                                V::Ok(len, ln) => {
                                    let payload = lp + ln;
                                    let end = match payload.checked_add(len as usize) {
                                        Some(x) => x,
                                        None => {
                                            if s.fin {
                                                Action::Error("TRUNCATED")
                                            } else {
                                                Action::Done
                                            }
                                        }
                                    };
                                    if end > s.buf.len() {
                                        if s.fin {
                                            Action::Error("TRUNCATED")
                                        } else {
                                            Action::Done
                                        }
                                    } else if typ == 3 {
                                        s.pos = end;
                                        Action::Connection
                                    } else {
                                        let pe = end;
                                        let ps = payload;

                                        match typ {
                                            0 => {
                                                if !s.final_headers || s.trailers {
                                                    Action::Error("FRAME")
                                                } else {
                                                    s.pos = pe;
                                                    s.next_number += 1;
                                                    Action::Frame(b'D')
                                                }
                                            }
                                            1 => {
                                                if ps == pe {
                                                    Action::Error("FRAME")
                                                } else {
                                                    let kind = s.buf[ps];
                                                    let rp = ps + 1;
                                                    if kind > 2 {
                                                        Action::Error("FRAME")
                                                    } else if rp == pe {
                                                        Action::Error("FRAME")
                                                    } else {
                                                        match varint(&s.buf, rp) {
                                                            V::Need => Action::Error("FRAME"),
                                                            V::Bad => Action::Error("VARINT"),
                                                            V::Ok(required, rn) => {
                                                                if rp + rn != pe {
                                                                    Action::Error("FRAME")
                                                                } else {
                                                                    let valid = match kind {
                                                                        0 => !s.final_headers && !s.trailers,
                                                                        1 => !s.final_headers && !s.trailers,
                                                                        2 => s.final_headers && !s.trailers,
                                                                        _ => false,
                                                                    };
                                                                    if !valid {
                                                                        Action::Error("FRAME")
                                                                    } else if required > *insertion_count {
                                                                        Action::Block(required)
                                                                    } else {
                                                                        s.pos = pe;
                                                                        s.next_number += 1;
                                                                        match kind {
                                                                            0 => Action::Frame(b'I'),
                                                                            1 => {
                                                                                s.final_headers = true;
                                                                                Action::Frame(b'F')
                                                                            }
                                                                            2 => {
                                                                                s.trailers = true;
                                                                                Action::Frame(b'T')
                                                                            }
                                                                            _ => Action::Error("FRAME"),
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            2 => {
                                                if ps == pe {
                                                    Action::Error("FRAME")
                                                } else {
                                                    match varint(&s.buf, ps) {
                                                        V::Need => Action::Error("FRAME"),
                                                        V::Bad => Action::Error("VARINT"),
                                                        V::Ok(delta, dn) => {
                                                            if ps + dn != pe || delta == 0 {
                                                                Action::Error("FRAME")
                                                            } else if *insertion_count + delta > MAX {
                                                                Action::Error("FRAME")
                                                            } else {
                                                                *insertion_count += delta;
                                                                s.pos = pe;
                                                                s.next_number += 1;
                                                                Action::Frame(b'C')
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {
                                                s.pos = pe;
                                                s.next_number += 1;
                                                Action::Frame(b'X')
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        match action {
            Action::Frame(label) => {
                if let Some(p) = pending.take() {
                    write_frame(out, &p, b'0')?;
                }
                let number = streams.get(&id).unwrap().next_number;
                pending = Some(Pending {
                    packet,
                    stream: id,
                    number,
                    label,
                });
            }
            Action::Block(req) => {
                if let Some(p) = pending.take() {
                    write_frame(out, &p, b'0')?;
                }
                if let Some(s) = streams.get_mut(&id) {
                    s.blocked = Some(req);
                }
                blocked_set.insert((req, id));
                return Ok(false);
            }
            Action::Error(reason) => {
                if let Some(p) = pending.take() {
                    write_frame(out, &p, b'0')?;
                }
                if let Some(s) = streams.get_mut(&id) {
                    s.dead = true;
                    s.blocked = None;
                }
                blocked_set.retain(|&(_, sid)| sid != id);
                write_stream_error(out, packet, id, reason)?;
                return Ok(false);
            }
            Action::Connection => {
                if let Some(p) = pending.take() {
                    write_frame(out, &p, b'0')?;
                }
                write_connection_error(out, packet, id)?;
                return Ok(true);
            }
            Action::Finish => {
                if let Some(p) = pending.take() {
                    write_frame(out, &p, b'1')?;
                }
                if let Some(s) = streams.get_mut(&id) {
                    s.closed = true;
                    s.blocked = None;
                }
                blocked_set.retain(|&(_, sid)| sid != id);
                return Ok(false);
            }
            Action::Done => {
                if let Some(p) = pending.take() {
                    write_frame(out, &p, b'0')?;
                }
                return Ok(false);
            }
        }
    }
}

fn main() -> io::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut it = input.split_whitespace();

    let n: usize = match it.next() {
        Some(x) => x.parse().unwrap(),
        None => return Ok(()),
    };

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let mut streams: BTreeMap<u64, Stream> = BTreeMap::new();
    let mut blocked_set: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut insertion_count = 0u64;
    let mut connection_error = false;

    for packet_no in 1..=n {
        let id: u64 = it.next().unwrap().parse().unwrap();
        let fin: u8 = it.next().unwrap().parse().unwrap();
        let hex = it.next().unwrap();
        let bytes = hex_bytes(hex);

        let should_process = {
            let s = streams.entry(id).or_insert_with(|| Stream {
                buf: Vec::new(),
                pos: 0,
                fin: false,
                final_headers: false,
                trailers: false,
                next_number: 0,
                blocked: None,
                dead: false,
                closed: false,
            });

            if s.dead || s.closed {
                false
            } else {
                s.buf.extend_from_slice(&bytes);
                if fin == 1 {
                    s.fin = true;
                }
                s.blocked.is_none()
            }
        };

        if should_process {
            if process_stream(
                id,
                packet_no as u64,
                &mut streams,
                &mut blocked_set,
                &mut insertion_count,
                &mut out,
            )? {
                connection_error = true;
                break;
            }
        }

        loop {
            let next = match blocked_set.iter().next().copied() {
                Some(x) if x.0 <= insertion_count => x,
                _ => break,
            };
            blocked_set.remove(&next);
            if let Some(s) = streams.get_mut(&next.1) {
                s.blocked = None;
            }
            if process_stream(
                next.1,
                packet_no as u64,
                &mut streams,
                &mut blocked_set,
                &mut insertion_count,
                &mut out,
            )? {
                connection_error = true;
                break;
            }
        }

        if connection_error {
            break;
        }
    }

    if !connection_error {
        let end_packet = n as u64 + 1;
        for (&id, s) in streams.iter() {
            if s.blocked.is_some() {
                write_stream_error(&mut out, end_packet, id, "BLOCKED")?;
            }
        }
    }

    out.flush()?;
    Ok(())
}