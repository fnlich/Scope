use std::io::{self,Read,Write};
use std::collections::{HashMap,BTreeSet};

struct Fen{t:Vec<i64>}
impl Fen{
    fn new(n:usize)->Fen{Fen{t:vec![0;n+2]}}
    fn add(&mut self,i:usize,v:i64){
        let mut i=i+1;
        while i<self.t.len(){ self.t[i]+=v; i+= i & i.wrapping_neg(); }
    }
    fn sum(&self,i:usize)->i64{
        let mut i=i+1; let mut s=0i64;
        while i>0 { s+=self.t[i]; i-= i & i.wrapping_neg(); }
        s
    }
}

fn pi64(s:&[u8])->i64{
    let mut i=0; let mut neg=false;
    if s[0]==b'-'{neg=true;i=1;} else if s[0]==b'+'{i=1;}
    let mut v:i128=0;
    while i<s.len(){ v=v*10+((s[i]-b'0') as i128); i+=1; }
    if neg { (-v) as i64 } else { v as i64 }
}

struct Group{ name:Vec<u8>, addrs:BTreeSet<i64>, prim:usize, exists:bool }

fn main(){
    let mut buf=Vec::new();
    io::stdin().read_to_end(&mut buf).unwrap();
    let toks:Vec<&[u8]>=buf.split(|c| matches!(c,0x09..=0x0d|0x20)).filter(|s| !s.is_empty()).collect();
    let mut tp=0usize;

    let mut gmap:HashMap<Vec<u8>,usize>=HashMap::new();
    let mut groups:Vec<Group>=Vec::new();
    let mut tmap:HashMap<Vec<u8>,usize>=HashMap::new();
    let mut ntopic=0usize;

    let r=pi64(toks[tp]) as usize; tp+=1;

    let mut rounds:Vec<(Vec<(usize,i64,bool)>,Vec<(usize,Vec<(i64,i64)>)>)>=Vec::with_capacity(r);
    let mut coords:Vec<i128>=Vec::new();

    for _ in 0..r {
        let d=pi64(toks[tp]) as usize; tp+=1;
        let mut discs:Vec<(usize,i64,bool)>=Vec::with_capacity(d);
        for _ in 0..d {
            let gname=toks[tp].to_vec(); tp+=1;
            let a=pi64(toks[tp]); tp+=1;
            let role=toks[tp][0]==b'P'; tp+=1;
            let id=match gmap.get(&gname){
                Some(&i)=>i,
                None=>{ let i=groups.len(); groups.push(Group{name:gname.clone(),addrs:BTreeSet::new(),prim:0,exists:false}); gmap.insert(gname,i); i }
            };
            coords.push(a as i128);
            discs.push((id,a,role));
        }
        let u=pi64(toks[tp]) as usize; tp+=1;
        let mut repls:Vec<(usize,Vec<(i64,i64)>)>=Vec::with_capacity(u);
        for _ in 0..u {
            let tname=toks[tp].to_vec(); tp+=1;
            let id=match tmap.get(&tname){ Some(&i)=>i, None=>{ let i=ntopic; ntopic+=1; tmap.insert(tname,i); i } };
            let k=pi64(toks[tp]) as usize; tp+=1;
            let mut iv:Vec<(i64,i64)>=Vec::with_capacity(k);
            for _ in 0..k {
                let l=pi64(toks[tp]); tp+=1;
                let rr=pi64(toks[tp]); tp+=1;
                coords.push(l as i128);
                coords.push((rr as i128)+1);
                iv.push((l,rr));
            }
            repls.push((id,iv));
        }
        rounds.push((discs,repls));
    }

    coords.sort();
    coords.dedup();
    let cn=coords.len();
    let idxof=|v:i128|->usize{
        match coords.binary_search(&v){ Ok(i)=>i, Err(i)=>i }
    };

    let mut fen=Fen::new(cn+2);
    let mut routes:Vec<Vec<(i64,i64)>>=vec![Vec::new();ntopic];
    let mut registry:HashMap<i64,(usize,bool)>=HashMap::new();
    let mut allset:BTreeSet<i64>=BTreeSet::new();
    let mut stamp:Vec<u32>=vec![u32::MAX;groups.len()];

    let out=io::stdout();
    let mut w=io::BufWriter::new(out.lock());

    for ri in 0..r {
        let round_id=ri as u32;
        let (discs,repls)=std::mem::replace(&mut rounds[ri],(Vec::new(),Vec::new()));
        let mut touched:Vec<usize>=Vec::new();
        let mut cand:Vec<i64>=Vec::new();

        for (g,a,role) in discs.into_iter() {
            if !groups[g].exists { groups[g].exists=true; }
            let prev=registry.get(&a).copied();
            match prev {
                Some((og,orole))=>{
                    if og!=g {
                        groups[og].addrs.remove(&a);
                        if orole { groups[og].prim-=1; }
                        if stamp[og]!=round_id { stamp[og]=round_id; touched.push(og); }
                        groups[g].addrs.insert(a);
                        if role { groups[g].prim+=1; }
                        registry.insert(a,(g,role));
                    } else {
                        if orole!=role {
                            if orole { groups[g].prim-=1; } else { groups[g].prim+=1; }
                            registry.insert(a,(g,role));
                        }
                    }
                },
                None=>{
                    registry.insert(a,(g,role));
                    groups[g].addrs.insert(a);
                    if role { groups[g].prim+=1; }
                    allset.insert(a);
                }
            }
            if stamp[g]!=round_id { stamp[g]=round_id; touched.push(g); }
            cand.push(a);
        }

        let mut removed_iv:Vec<(i64,i64)>=Vec::new();
        for (t,niv) in repls.into_iter() {
            let old=std::mem::replace(&mut routes[t],Vec::new());
            for &(l,rr) in old.iter() {
                fen.add(idxof(l as i128),-1);
                fen.add(idxof((rr as i128)+1),1);
                removed_iv.push((l,rr));
            }
            for &(l,rr) in niv.iter() {
                fen.add(idxof(l as i128),1);
                fen.add(idxof((rr as i128)+1),-1);
            }
            routes[t]=niv;
        }

        for &(l,rr) in removed_iv.iter() {
            for &a in allset.range(l..=rr) { cand.push(a); }
        }

        cand.sort();
        cand.dedup();

        let mut removals:Vec<(usize,i64)>=Vec::new();

        for &a in cand.iter() {
            if let Some(&(g,role))=registry.get(&a) {
                if fen.sum(idxof(a as i128))<=0 {
                    registry.remove(&a);
                    groups[g].addrs.remove(&a);
                    if role { groups[g].prim-=1; }
                    allset.remove(&a);
                    if stamp[g]!=round_id { stamp[g]=round_id; touched.push(g); }
                    removals.push((g,a));
                }
            }
        }

        let tl=touched.clone();
        for &g in tl.iter() {
            if groups[g].exists && groups[g].prim==0 && !groups[g].addrs.is_empty() {
                let set=std::mem::replace(&mut groups[g].addrs,BTreeSet::new());
                for a in set.into_iter() {
                    registry.remove(&a);
                    allset.remove(&a);
                    removals.push((g,a));
                }
                groups[g].prim=0;
            }
        }

        let mut deleted:Vec<usize>=Vec::new();
        for &g in tl.iter() {
            if groups[g].exists && groups[g].addrs.is_empty() {
                groups[g].exists=false;
                groups[g].prim=0;
                deleted.push(g);
            }
        }

        removals.sort_by(|x,y|{
            let c=groups[x.0].name.cmp(&groups[y.0].name);
            if c==std::cmp::Ordering::Equal { x.1.cmp(&y.1) } else { c }
        });
        deleted.sort_by(|&x,&y| groups[x].name.cmp(&groups[y].name));

        let mut line:Vec<u8>=Vec::new();
        line.extend_from_slice(removals.len().to_string().as_bytes());
        for &(g,a) in removals.iter() {
            line.push(b' ');
            line.extend_from_slice(&groups[g].name);
            line.push(b' ');
            line.extend_from_slice(a.to_string().as_bytes());
        }
        line.push(b' ');
        line.extend_from_slice(deleted.len().to_string().as_bytes());
        for &g in deleted.iter() {
            line.push(b' ');
            line.extend_from_slice(&groups[g].name);
        }
        line.push(b'\n');
        w.write_all(&line).unwrap();
    }
    w.flush().unwrap();
}