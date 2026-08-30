use std::io::{self,Read,Write};

fn findn(a:&mut Vec<u32>, x:u32)->u32{
    let mut r=x;
    while a[r as usize]!=r { r=a[r as usize]; }
    let mut c=x;
    while a[c as usize]!=c { let n=a[c as usize]; a[c as usize]=r; c=n; }
    r
}

fn main(){
    let mut data=Vec::new();
    io::stdin().read_to_end(&mut data).unwrap();
    let n=data.len();
    let mut toks:Vec<(u32,u32)>=Vec::with_capacity(1024);
    let mut i=0usize;
    while i<n {
        let c=data[i];
        if c==b' '||(c>=0x09&&c<=0x0D) { i+=1; continue; }
        let s=i;
        while i<n {
            let c=data[i];
            if c==b' '||(c>=0x09&&c<=0x0D) { break; }
            i+=1;
        }
        toks.push((s as u32,i as u32));
    }
    let tget=|t:usize|->&[u8]{ let (a,b)=toks[t]; &data[a as usize..b as usize] };
    let tnum=|t:usize|->usize{ let (a,b)=toks[t]; let mut v=0usize; let mut k=a as usize; while k<b as usize { v=v*10+(data[k]-b'0') as usize; k+=1; } v };

    let mut p=0usize;
    let s_count=tnum(p); p+=1;
    let mut key_map:std::collections::HashMap<&[u8],u32>=std::collections::HashMap::new();
    let mut key_tok:Vec<u32>=Vec::new();
    let mut entries:Vec<(u32,u32)>=Vec::new();
    let mut trange:Vec<(u32,u32)>=Vec::with_capacity(s_count+1);
    trange.push((0,0));
    for _ in 0..s_count {
        let m=tnum(p); p+=1;
        let st=entries.len() as u32;
        for _ in 0..m {
            let kt=p; p+=1;
            let vt=p; p+=1;
            let kb=tget(kt);
            let id = match key_map.get(kb) {
                Some(&x)=>x,
                None=>{ let x=key_tok.len() as u32; key_tok.push(kt as u32); key_map.insert(kb,x); x }
            };
            entries.push((id,vt as u32));
        }
        trange.push((st,entries.len() as u32));
    }
    let q=tnum(p); p+=1;
    let mut cmd_kind:Vec<u8>=Vec::with_capacity(q);
    let mut ca:Vec<u32>=Vec::with_capacity(q);
    let mut cb:Vec<u32>=Vec::with_capacity(q);
    let mut cm:Vec<u8>=Vec::with_capacity(q);
    let mut mode:u8=0;
    for _ in 0..q {
        let w=tget(p);
        if w==b"FLIP" { p+=1; mode^=1; cmd_kind.push(2); ca.push(0); cb.push(0); cm.push(mode); }
        else if w==b"SET" {
            p+=1;
            let kt=p; p+=1;
            let vt=p; p+=1;
            let kb=tget(kt);
            let id = match key_map.get(kb) {
                Some(&x)=>x,
                None=>{ let x=key_tok.len() as u32; key_tok.push(kt as u32); key_map.insert(kb,x); x }
            };
            cmd_kind.push(0); ca.push(id); cb.push(vt as u32); cm.push(mode);
        } else {
            p+=1;
            let l=tnum(p); p+=1;
            let r=tnum(p); p+=1;
            cmd_kind.push(1); ca.push(l as u32); cb.push(r as u32); cm.push(mode);
        }
    }

    let nk=key_tok.len();
    let mut inserted=vec![false;nk];
    let mut order:Vec<u32>=Vec::with_capacity(nk);
    let sc=s_count as u32;
    let mut nxt:Vec<u32>=(0..sc+2).collect();
    let mut prv:Vec<u32>=(0..sc+2).collect();
    for c in 0..q {
        if cmd_kind[c]==0 {
            let id=ca[c] as usize;
            if !inserted[id] { inserted[id]=true; order.push(id as u32); }
        } else if cmd_kind[c]==1 {
            let l=ca[c]; let r=cb[c];
            if cm[c]==0 {
                let mut j=findn(&mut nxt,l);
                while j<=r {
                    let (a,b)=trange[j as usize];
                    for e in a..b {
                        let id=entries[e as usize].0 as usize;
                        if !inserted[id] { inserted[id]=true; order.push(id as u32); }
                    }
                    nxt[j as usize]=j+1; prv[j as usize]=j-1;
                    j=findn(&mut nxt,j+1);
                }
            } else {
                let mut j=findn(&mut prv,r);
                while j>=l && j>=1 {
                    let (a,b)=trange[j as usize];
                    for e in a..b {
                        let id=entries[e as usize].0 as usize;
                        if !inserted[id] { inserted[id]=true; order.push(id as u32); }
                    }
                    nxt[j as usize]=j+1; prv[j as usize]=j-1;
                    if j==0 { break; }
                    j=findn(&mut prv,j-1);
                }
            }
        }
    }

    let mut val:Vec<u32>=vec![u32::MAX;nk];
    let mut nxt2:Vec<u32>=(0..sc+2).collect();
    let mut prv2:Vec<u32>=(0..sc+2).collect();
    for ci in (0..q).rev() {
        if cmd_kind[ci]==0 {
            let id=ca[ci] as usize;
            if val[id]==u32::MAX { val[id]=cb[ci]; }
        } else if cmd_kind[ci]==1 {
            let l=ca[ci]; let r=cb[ci];
            if cm[ci]==0 {
                let mut j=findn(&mut prv2,r);
                while j>=l && j>=1 {
                    let (a,b)=trange[j as usize];
                    for e in a..b {
                        let (id,vt)=entries[e as usize];
                        if val[id as usize]==u32::MAX { val[id as usize]=vt; }
                    }
                    nxt2[j as usize]=j+1; prv2[j as usize]=j-1;
                    if j==0 { break; }
                    j=findn(&mut prv2,j-1);
                }
            } else {
                let mut j=findn(&mut nxt2,l);
                while j<=r {
                    let (a,b)=trange[j as usize];
                    for e in a..b {
                        let (id,vt)=entries[e as usize];
                        if val[id as usize]==u32::MAX { val[id as usize]=vt; }
                    }
                    nxt2[j as usize]=j+1; prv2[j as usize]=j-1;
                    j=findn(&mut nxt2,j+1);
                }
            }
        }
    }

    let so=io::stdout();
    let mut out=io::BufWriter::new(so.lock());
    let mut buf:Vec<u8>=Vec::with_capacity(1<<16);
    buf.extend_from_slice(order.len().to_string().as_bytes());
    buf.push(b'\n');
    for &id in order.iter() {
        let kt=key_tok[id as usize] as usize;
        let (a,b)=toks[kt];
        buf.extend_from_slice(&data[a as usize..b as usize]);
        buf.push(b' ');
        let v=val[id as usize];
        if v==u32::MAX { buf.push(b'-'); }
        else {
            let (a2,b2)=toks[v as usize];
            buf.extend_from_slice(&data[a2 as usize..b2 as usize]);
        }
        buf.push(b'\n');
    }
    out.write_all(&buf).unwrap();
}