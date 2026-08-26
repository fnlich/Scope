use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Transform {
    r: u8,
    x: i64,
    y: i64,
}

impl Transform {
    fn id() -> Self {
        Self { r: 0, x: 0, y: 0 }
    }

    fn apply_point(self, x: i64, y: i64) -> (i64, i64) {
        let (rx, ry) = match self.r {
            0 => (x, y),
            1 => (-y, x),
            2 => (-x, -y),
            _ => (y, -x),
        };
        (rx + self.x, ry + self.y)
    }

    fn inverse(self) -> Self {
        let r = (4 - self.r) & 3;
        let (x, y) = match r {
            0 => (self.x, self.y),
            1 => (-self.y, self.x),
            2 => (-self.x, -self.y),
            _ => (self.y, -self.x),
        };
        Self {
            r,
            x: -x,
            y: -y,
        }
    }
}

fn compose(a: Transform, b: Transform) -> Transform {
    let (x, y) = a.apply_point(b.x, b.y);
    Transform {
        r: (a.r + b.r) & 3,
        x,
        y,
    }
}

fn parse_id(s: &str) -> Option<u32> {
    if s.is_empty() || (s.len() > 1 && s.as_bytes()[0] == b'0') {
        return None;
    }
    let mut v = 0u32;
    for b in s.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add((b - b'0') as u32)?;
        if v > 1_000_000_000 {
            return None;
        }
    }
    Some(v)
}

fn parse_t(s: &str) -> Option<u8> {
    match s {
        "color" => Some(0),
        "depth" => Some(1),
        "gyro" => Some(2),
        _ => None,
    }
}

fn parse_p(s: &str) -> Option<u8> {
    match s {
        "image" => Some(0),
        "range" => Some(1),
        "motion" => Some(2),
        _ => None,
    }
}

fn parse_o(s: &str) -> Option<u8> {
    match s {
        "gain" => Some(0),
        "exposure" => Some(1),
        _ => None,
    }
}

enum Card {
    Device(u32),
    Sensor(u32, u32),
    Stream(u32, u32, u8, u32),
    Frame(u32, u32, u8, u32, u8),
    Option(u32, u32, u8, u32, u8),
    Calibration(u32, u32, u8, u32, u32, u32, u8, u32),
}

fn parse_path(path: &str) -> Option<Card> {
    let s: Vec<&str> = path.split('/').collect();
    if s.len() < 4 || s[0] != "table" || s[1] != "v1" || s[2] != "device" {
        return None;
    }

    match s.len() {
        4 => {
            let d = parse_id(s[3])?;
            Some(Card::Device(d))
        }
        6 => {
            if s[4] != "sensor" {
                return None;
            }
            let d = parse_id(s[3])?;
            let sid = parse_id(s[5])?;
            Some(Card::Sensor(d, sid))
        }
        8 => {
            if s[4] != "sensor" || s[6] != "stream" {
                return None;
            }
            let d = parse_id(s[3])?;
            let sid = parse_id(s[5])?;
            let t = parse_t(s[7])?;
            None.or_else(|| {
                let _ = t;
                Some(Card::Stream(d, sid, t, 0))
            })
        }
        10 => {
            if s[4] != "sensor" || s[6] != "stream" {
                return None;
            }
            let d = parse_id(s[3])?;
            let sid = parse_id(s[5])?;
            let t = parse_t(s[7])?;
            let i = parse_id(s[8])?;
            match s[9] {
                "frame" => {
                    let p = parse_p(s[9]).or_else(|| parse_p(s[9]))?;
                    let _ = p;
                    None
                }
                _ => None,
            }
            .or_else(|| {
                if s[9] == "frame" {
                    None
                } else {
                    None
                }
            })?;
            unreachable!()
        }
        16 => {
            if s[4] != "sensor"
                || s[6] != "stream"
                || s[9] != "calibration"
                || s[10] != "device"
                || s[12] != "sensor"
                || s[14] != "stream"
            {
                return None;
            }
            let d = parse_id(s[3])?;
            let sid = parse_id(s[5])?;
            let t = parse_t(s[7])?;
            let i = parse_id(s[8])?;
            let d2 = parse_id(s[11])?;
            let sid2 = parse_id(s[13])?;
            let t2 = parse_t(s[15])?;
            let i2 = parse_id(s[16])?;
            Some(Card::Calibration(d, sid, t, i, d2, sid2, t2, i2))
        }
        _ => None,
    }
}

fn parse_path_fixed(path: &str) -> Option<Card> {
    let s: Vec<&str> = path.split('/').collect();
    if s.len() < 4 || s[0] != "table" || s[1] != "v1" || s[2] != "device" {
        return None;
    }

    let d = parse_id(s[3])?;

    match s.len() {
        4 => Some(Card::Device(d)),
        6 => {
            if s[4] != "sensor" {
                None
            } else {
                Some(Card::Sensor(d, parse_id(s[5])?))
            }
        }
        8 => {
            if s[4] != "sensor" || s[6] != "stream" {
                return None;
            }
            Some(Card::Stream(
                d,
                parse_id(s[5])?,
                parse_t(s[7])?,
                0,
            ))
        }
        10 => {
            if s[4] != "sensor" || s[6] != "stream" {
                return None;
            }
            let sid = parse_id(s[5])?;
            let t = parse_t(s[7])?;
            let i = parse_id(s[8])?;
            match s[9] {
                "frame" => {
                    if s.len() != 11 {
                        None
                    } else {
                        Some(Card::Frame(d, sid, t, i, parse_p(s[10])?))
                    }
                }
                "option" => {
                    if s.len() != 11 {
                        None
                    } else {
                        Some(Card::Option(d, sid, t, i, parse_o(s[10])?))
                    }
                }
                _ => None,
            }
        }
        16 => {
            if s[4] != "sensor"
                || s[6] != "stream"
                || s[9] != "calibration"
                || s[10] != "device"
                || s[12] != "sensor"
                || s[14] != "stream"
            {
                return None;
            }
            Some(Card::Calibration(
                d,
                parse_id(s[5])?,
                parse_t(s[7])?,
                parse_id(s[8])?,
                parse_id(s[11])?,
                parse_id(s[13])?,
                parse_t(s[15])?,
                parse_id(s[16])?,
            ))
        }
        _ => None,
    }
}

struct Dsu {
    parent: Vec<usize>,
    pot: Vec<Transform>,
    obs: Vec<HashMap<i64, (i64, i64)>>,
}

impl Dsu {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            pot: Vec::new(),
            obs: Vec::new(),
        }
    }

    fn add(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.pot.push(Transform::id());
        self.obs.push(HashMap::new());
        id
    }

    fn find(&mut self, x: usize) -> (usize, Transform) {
        let mut nodes = Vec::new();
        let mut cur = x;
        while self.parent[cur] != cur {
            nodes.push(cur);
            cur = self.parent[cur];
        }
        let root = cur;

        let mut parent_to_root = Transform::id();
        for &node in nodes.iter().rev() {
            let old = self.pot[node];
            let node_to_root = compose(parent_to_root, old);
            self.parent[node] = root;
            self.pot[node] = node_to_root;
            parent_to_root = node_to_root;
        }

        (root, self.pot[x])
    }

    fn add_frame(
        &mut self,
        stream: usize,
        timestamp: i64,
        x: i64,
        y: i64,
    ) -> bool {
        let (root, tr) = self.find(stream);
        let p = tr.apply_point(x, y);
        if let Some(&old) = self.obs[root].get(&timestamp) {
            old == p
        } else {
            self.obs[root].insert(timestamp, p);
            true
        }
    }

    fn add_calibration(
        &mut self,
        source: usize,
        target: usize,
        tr: Transform,
    ) -> Result<bool, ()> {
        let (rs, a) = self.find(source);
        let (rt, b) = self.find(target);

        if rs == rt {
            let implied = compose(b.inverse(), a);
            return Ok(implied == tr);
        }

        let source_to_target_roots = compose(b, compose(tr, a.inverse()));

        let source_size = self.obs[rs].len();
        let target_size = self.obs[rt].len();

        let (child, parent, child_to_parent) = if source_size <= target_size {
            (rs, rt, source_to_target_roots)
        } else {
            (rt, rs, source_to_target_roots.inverse())
        };

        self.parent[child] = parent;
        self.pot[child] = child_to_parent;

        let mismatch = if child < parent {
            let (left, right) = self.obs.split_at_mut(parent);
            let child_map = &mut left[child];
            let parent_map = &mut right[0];
            let mut bad = false;
            for (ts, p) in child_map.drain() {
                let np = child_to_parent.apply_point(p.0, p.1);
                if let Some(&old) = parent_map.get(&ts) {
                    if old != np {
                        bad = true;
                        break;
                    }
                } else {
                    parent_map.insert(ts, np);
                }
            }
            bad
        } else {
            let (left, right) = self.obs.split_at_mut(child);
            let parent_map = &mut left[parent];
            let child_map = &mut right[0];
            let mut bad = false;
            for (ts, p) in child_map.drain() {
                let np = child_to_parent.apply_point(p.0, p.1);
                if let Some(&old) = parent_map.get(&ts) {
                    if old != np {
                        bad = true;
                        break;
                    }
                } else {
                    parent_map.insert(ts, np);
                }
            }
            bad
        };

        if mismatch {
            Err(())
        } else {
            Ok(true)
        }
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let n: usize = it.next().unwrap().parse().unwrap();

    let mut devices: HashSet<u32> = HashSet::new();
    let mut sensors: HashSet<(u32, u32)> = HashSet::new();
    let mut streams: HashMap<(u32, u32, u8, u32), usize> = HashMap::new();
    let mut options: HashSet<(usize, u8)> = HashSet::new();

    let mut dsu = Dsu::new();
    let mut next_q: Vec<i128> = Vec::new();
    let mut last_t: Vec<Option<i64>> = Vec::new();

    let mut islands = 0usize;
    let mut invalid: Option<(usize, &'static str)> = None;

    for k in 1..=n {
        let t: i64 = it.next().unwrap().parse().unwrap();
        let path = it.next().unwrap();
        let q: i64 = it.next().unwrap().parse().unwrap();
        let x: i64 = it.next().unwrap().parse().unwrap();
        let y: i64 = it.next().unwrap().parse().unwrap();

        let card = match parse_path_fixed(path) {
            Some(c) => c,
            None => {
                invalid = Some((k, "PATH"));
                break;
            }
        };

        let fail = match card {
            Card::Device(d) => {
                if t != -1 {
                    Some("TIME")
                } else if q != 0 || x != 0 || y != 0 {
                    Some("PAYLOAD")
                } else if devices.contains(&d) {
                    Some("DUPLICATE")
                } else {
                    devices.insert(d);
                    None
                }
            }

            Card::Sensor(d, s) => {
                if t != -1 {
                    Some("TIME")
                } else if q != 0 || x != 0 || y != 0 {
                    Some("PAYLOAD")
                } else if !devices.contains(&d) {
                    Some("ORDER")
                } else if sensors.contains(&(d, s)) {
                    Some("DUPLICATE")
                } else {
                    sensors.insert((d, s));
                    None
                }
            }

            Card::Stream(d, s, ty, i) => {
                if t != -1 {
                    Some("TIME")
                } else if q != 0 || x != 0 || y != 0 {
                    Some("PAYLOAD")
                } else if !sensors.contains(&(d, s)) {
                    Some("ORDER")
                } else if streams.contains_key(&(d, s, ty, i)) {
                    Some("DUPLICATE")
                } else {
                    let id = dsu.add();
                    streams.insert((d, s, ty, i), id);
                    next_q.push(0);
                    last_t.push(None);
                    islands += 1;
                    None
                }
            }

            Card::Frame(d, s, ty, i, p) => {
                if t < 0 {
                    Some("TIME")
                } else if q < 0 {
                    Some("PAYLOAD")
                } else {
                    match streams.get(&(d, s, ty, i)).copied() {
                        None => Some("ORDER"),
                        Some(id) => {
                            let expected_p = match ty {
                                0 => 0,
                                1 => 1,
                                _ => 2,
                            };
                            if p != expected_p {
                                Some("MAPPING")
                            } else if q as i128 != next_q[id] {
                                Some("SEQUENCE")
                            } else if let Some(prev) = last_t[id] {
                                if t <= prev {
                                    Some("SEQUENCE")
                                } else {
                                    next_q[id] = q as i128 + 1;
                                    last_t[id] = Some(t);
                                    if dsu.add_frame(id, t, x, y) {
                                        None
                                    } else {
                                        Some("OBSERVATION")
                                    }
                                }
                            } else {
                                next_q[id] = q as i128 + 1;
                                last_t[id] = Some(t);
                                if dsu.add_frame(id, t, x, y) {
                                    None
                                } else {
                                    Some("OBSERVATION")
                                }
                            }
                        }
                    }
                }
            }

            Card::Option(d, s, ty, i, o) => {
                if t != -1 {
                    Some("TIME")
                } else if !(0..=100).contains(&q) || x != 0 || y != 0 {
                    Some("PAYLOAD")
                } else {
                    match streams.get(&(d, s, ty, i)).copied() {
                        None => Some("ORDER"),
                        Some(id) => {
                            if o == 1 && ty == 2 {
                                Some("MAPPING")
                            } else if options.contains(&(id, o)) {
                                Some("DUPLICATE")
                            } else {
                                options.insert((id, o));
                                None
                            }
                        }
                    }
                }
            }

            Card::Calibration(d, s, ty, i, d2, s2, ty2, i2) => {
                if t != -1 {
                    Some("TIME")
                } else if !(0..=3).contains(&q) {
                    Some("PAYLOAD")
                } else {
                    let source = streams.get(&(d, s, ty, i)).copied();
                    let target = streams.get(&(d2, s2, ty2, i2)).copied();

                    match (source, target) {
                        (Some(a), Some(b)) => {
                            let tr = Transform {
                                r: q as u8,
                                x,
                                y,
                            };
                            match dsu.add_calibration(a, b, tr) {
                                Ok(true) => {
                                    let (ra, _) = dsu.find(a);
                                    let (rb, _) = dsu.find(b);
                                    if ra != rb {
                                        islands -= 1;
                                    }
                                    None
                                }
                                Ok(false) => Some("CALIBRATION"),
                                Err(()) => Some("OBSERVATION"),
                            }
                        }
                        _ => Some("ORDER"),
                    }
                }
            }
        };

        if let Some(code) = fail {
            invalid = Some((k, code));
            break;
        }
    }

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    if let Some((k, code)) = invalid {
        writeln!(out, "INVALID {} {}", k, code).unwrap();
    } else {
        writeln!(out, "VALID {}", islands).unwrap();
    }
}