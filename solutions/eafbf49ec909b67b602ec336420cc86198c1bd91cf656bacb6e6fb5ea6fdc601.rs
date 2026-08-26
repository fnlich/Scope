use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};

struct Node {
    key: u64,
    val: i64,
    next: Option<usize>,
    prev: Option<usize>,
    table: usize,
    bucket: u64,
    active: bool,
}

struct Table {
    cap: u64,
    buckets: BTreeMap<u64, usize>,
    entries: usize,
}

impl Table {
    fn new(cap: u64) -> Self {
        Self {
            cap,
            buckets: BTreeMap::new(),
            entries: 0,
        }
    }

    fn bucket(&self, key: u64) -> u64 {
        key & (self.cap - 1)
    }
}

struct Iter {
    safe: bool,
    snap: Vec<usize>,
    pos: usize,
    rev: u64,
    invalid: bool,
}

struct Sim {
    nodes: Vec<Node>,
    map: HashMap<u64, usize>,
    tables: [Table; 2],
    active: usize,
    resizing: bool,
    cursor: u64,
    rev: u64,
    iter: Option<Iter>,
}

impl Sim {
    fn new(cap: u64) -> Self {
        Self {
            nodes: Vec::new(),
            map: HashMap::new(),
            tables: [Table::new(cap), Table::new(1)],
            active: 0,
            resizing: false,
            cursor: 0,
            rev: 0,
            iter: None,
        }
    }

    fn old_table(&self) -> usize {
        self.active
    }

    fn new_table(&self) -> usize {
        1 - self.active
    }

    fn insert_head(&mut self, tid: usize, id: usize, bucket: u64) {
        let old = self.tables[tid].buckets.get(&bucket).copied();

        self.nodes[id].table = tid;
        self.nodes[id].bucket = bucket;
        self.nodes[id].prev = None;
        self.nodes[id].next = old;

        if let Some(old_id) = old {
            self.nodes[old_id].prev = Some(id);
        }

        self.tables[tid].buckets.insert(bucket, id);
        self.tables[tid].entries += 1;
    }

    fn unlink(&mut self, id: usize) {
        let tid = self.nodes[id].table;
        let bucket = self.nodes[id].bucket;
        let prev = self.nodes[id].prev;
        let next = self.nodes[id].next;

        if let Some(prev_id) = prev {
            self.nodes[prev_id].next = next;
        } else {
            match next {
                Some(next_id) => {
                    self.tables[tid].buckets.insert(bucket, next_id);
                }
                None => {
                    self.tables[tid].buckets.remove(&bucket);
                }
            }
        }

        if let Some(next_id) = next {
            self.nodes[next_id].prev = prev;
        }

        self.nodes[id].prev = None;
        self.nodes[id].next = None;
        self.tables[tid].entries -= 1;
    }

    fn move_node(&mut self, id: usize, dest: usize) {
        let key = self.nodes[id].key;
        self.unlink(id);
        let bucket = self.tables[dest].bucket(key);
        self.insert_head(dest, id, bucket);
    }

    fn find(&self, key: u64) -> Option<usize> {
        self.map.get(&key).copied().filter(|&id| self.nodes[id].active)
    }

    fn finish_resize(&mut self) {
        if self.resizing {
            let old = self.old_table();
            self.tables[old].buckets.clear();
            self.tables[old].entries = 0;
            self.active = self.new_table();
            self.resizing = false;
            self.cursor = 0;
            self.rev += 1;
        }
    }

    fn maintenance(&mut self, w: u64) {
        if !self.resizing || w == 0 {
            return;
        }

        let inspection_limit = w.saturating_mul(4);
        if inspection_limit == 0 {
            return;
        }

        let old = self.old_table();
        let new = self.new_table();
        let cap = self.tables[old].cap;

        let mut inspected = 0u64;
        let mut migrated = 0u64;
        let mut changed = false;

        while inspected < inspection_limit && migrated < w && self.resizing {
            if self.cursor >= cap {
                if self.tables[old].entries == 0 {
                    self.finish_resize();
                    changed = true;
                }
                break;
            }

            let next_bucket = self.tables[old]
                .buckets
                .range(self.cursor..)
                .next()
                .map(|(&b, _)| b);

            match next_bucket {
                None => {
                    let available = cap - self.cursor;
                    let remaining = inspection_limit - inspected;
                    let take = available.min(remaining);
                    self.cursor += take;
                    inspected += take;
                    if take > 0 {
                        changed = true;
                    }

                    if self.cursor >= cap && self.tables[old].entries == 0 {
                        self.finish_resize();
                        changed = true;
                    }
                }
                Some(bucket) => {
                    let gap = bucket - self.cursor;
                    let cost = gap + 1;
                    let remaining = inspection_limit - inspected;

                    if cost > remaining {
                        self.cursor += remaining;
                        inspected += remaining;
                        if remaining > 0 {
                            changed = true;
                        }
                    } else {
                        inspected += cost;
                        self.cursor = bucket + 1;
                        migrated += 1;
                        changed = true;

                        let mut cur = self.tables[old].buckets.get(&bucket).copied();

                        while let Some(id) = cur {
                            let next = self.nodes[id].next;
                            self.move_node(id, new);
                            cur = next;
                        }

                        if self.tables[old].entries == 0 {
                            self.finish_resize();
                            changed = true;
                        }
                    }
                }
            }
        }

        if changed && self.resizing {
            self.rev += 1;
        }
    }

    fn put(&mut self, key: u64, value: i64) {
        if let Some(id) = self.find(key) {
            self.nodes[id].val = value;
            return;
        }

        let id = self.nodes.len();
        self.nodes.push(Node {
            key,
            val: value,
            next: None,
            prev: None,
            table: self.active,
            bucket: 0,
            active: true,
        });
        self.map.insert(key, id);

        let tid = if self.resizing {
            self.new_table()
        } else {
            self.active
        };

        let bucket = self.tables[tid].bucket(key);
        self.insert_head(tid, id, bucket);
        self.rev += 1;
    }

    fn get(&self, key: u64) -> Option<i64> {
        self.find(key).map(|id| self.nodes[id].val)
    }

    fn del(&mut self, key: u64) -> Option<i64> {
        let id = self.find(key)?;
        let value = self.nodes[id].val;

        self.unlink(id);
        self.nodes[id].active = false;
        self.map.remove(&key);
        self.rev += 1;

        Some(value)
    }

    fn resize(&mut self, cap: u64) {
        let dest = self.new_table();
        self.tables[dest] = Table::new(cap);
        self.cursor = 0;
        self.resizing = true;
        self.rev += 1;
    }

    fn begin(&mut self, safe: bool) {
        let mut snap = Vec::new();

        if self.resizing {
            let new = self.new_table();
            let old = self.old_table();

            for &head in self.tables[new].buckets.values() {
                let mut cur = Some(head);
                while let Some(id) = cur {
                    snap.push(id);
                    cur = self.nodes[id].next;
                }
            }

            for &head in self.tables[old].buckets.values() {
                let mut cur = Some(head);
                while let Some(id) = cur {
                    snap.push(id);
                    cur = self.nodes[id].next;
                }
            }
        } else {
            let tid = self.active;

            for &head in self.tables[tid].buckets.values() {
                let mut cur = Some(head);
                while let Some(id) = cur {
                    snap.push(id);
                    cur = self.nodes[id].next;
                }
            }
        }

        self.iter = Some(Iter {
            safe,
            snap,
            pos: 0,
            rev: self.rev,
            invalid: false,
        });
    }

    fn next(&mut self) -> String {
        let (safe, rev, invalid) = {
            let it = self.iter.as_ref().unwrap();
            (it.safe, it.rev, it.invalid)
        };

        if !safe {
            if invalid || rev != self.rev {
                self.iter.as_mut().unwrap().invalid = true;
                return "INVALID".to_string();
            }

            let (id, done) = {
                let it = self.iter.as_mut().unwrap();
                if it.pos >= it.snap.len() {
                    (0, true)
                } else {
                    let id = it.snap[it.pos];
                    it.pos += 1;
                    (id, false)
                }
            };

            if done {
                return "END".to_string();
            }

            return format!("ENTRY {} {}", self.nodes[id].key, self.nodes[id].val);
        }

        loop {
            let id = {
                let it = self.iter.as_mut().unwrap();
                if it.pos >= it.snap.len() {
                    return "END".to_string();
                }
                let id = it.snap[it.pos];
                it.pos += 1;
                id
            };

            if self.nodes[id].active {
                return format!("ENTRY {} {}", self.nodes[id].key, self.nodes[id].val);
            }
        }
    }

    fn print_table(&self, tid: usize, label: &str, out: &mut String) {
        let table = &self.tables[tid];
        out.push_str(&format!(
            "{} {} {} {}",
            label,
            table.cap,
            table.entries,
            table.buckets.len()
        ));

        for (&bucket, &head) in table.buckets.iter() {
            let mut len = 0usize;
            let mut cur = Some(head);

            while let Some(id) = cur {
                len += 1;
                cur = self.nodes[id].next;
            }

            out.push_str(&format!(" bucket {}", len));

            cur = Some(head);
            while let Some(id) = cur {
                out.push_str(&format!(" {} {}", self.nodes[id].key, self.nodes[id].val));
                cur = self.nodes[id].next;
            }

            let _ = bucket;
        }
    }

    fn output(&self, out: &mut String) {
        if !self.resizing {
            out.push_str("FINAL 0\n");
            self.print_table(self.active, "TABLE", out);
            out.push('\n');
        } else {
            out.push_str("FINAL 1\n");
            self.print_table(self.old_table(), "OLD", out);
            out.push('\n');
            self.print_table(self.new_table(), "NEW", out);
            out.push('\n');
        }
    }
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut it = input.split_whitespace();

    let initial_capacity: u64 = it.next().unwrap().parse().unwrap();
    let q: usize = it.next().unwrap().parse().unwrap();

    let mut sim = Sim::new(initial_capacity);
    let mut output = String::new();

    for _ in 0..q {
        let w: u64 = it.next().unwrap().parse().unwrap();
        let cmd = it.next().unwrap();

        if sim.iter.as_ref().map_or(true, |iter| !iter.safe) {
            sim.maintenance(w);
        }

        match cmd {
            "PUT" => {
                let key: u64 = it.next().unwrap().parse().unwrap();
                let value: i64 = it.next().unwrap().parse().unwrap();
                sim.put(key, value);
            }
            "GET" => {
                let key: u64 = it.next().unwrap().parse().unwrap();
                match sim.get(key) {
                    Some(value) => {
                        output.push_str(&format!("VALUE {}\n", value));
                    }
                    None => output.push_str("NONE\n"),
                }
            }
            "DEL" => {
                let key: u64 = it.next().unwrap().parse().unwrap();
                match sim.del(key) {
                    Some(value) => {
                        output.push_str(&format!("VALUE {}\n", value));
                    }
                    None => output.push_str("NONE\n"),
                }
            }
            "RESIZE" => {
                let capacity: u64 = it.next().unwrap().parse().unwrap();
                sim.resize(capacity);
            }
            "BEGIN" => {
                let mode = it.next().unwrap();
                sim.begin(mode == "SAFE");
            }
            "NEXT" => {
                output.push_str(&sim.next());
                output.push('\n');
            }
            "STOP" => {
                sim.iter = None;
            }
            _ => unreachable!(),
        }
    }

    sim.output(&mut output);

    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());
    writer.write_all(output.as_bytes()).unwrap();
}