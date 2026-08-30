use std::io::{self,Read,Write};
use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn safe(p:&str)->bool{
    if p.is_empty(){return false;}
    for c in p.split('/'){
        if c.is_empty(){return false;}
        if c=="."||c==".."{return false;}
        for b in c.bytes(){
            if !(b.is_ascii_alphanumeric()||b==b'.'||b==b'_'||b==b'-'){return false;}
        }
    }
    true
}

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it=s.split_ascii_whitespace();
    let out=io::stdout();
    let mut w=io::BufWriter::new(out.lock());
    let n:usize=match it.next(){Some(x)=>x.parse().unwrap(),None=>{return;}};
    let p:usize=it.next().unwrap().parse().unwrap();
    let b:i64=it.next().unwrap().parse().unwrap();
    let limit:i64=it.next().unwrap().parse().unwrap();
    let mut paths:Vec<&str>=Vec::with_capacity(n);
    let mut sizes:Vec<i64>=Vec::with_capacity(n);
    let mut digs:Vec<&str>=Vec::with_capacity(n);
    let mut durs:Vec<i64>=Vec::with_capacity(n);
    for _ in 0..n{
        paths.push(it.next().unwrap());
        sizes.push(it.next().unwrap().parse().unwrap());
        digs.push(it.next().unwrap());
        durs.push(it.next().unwrap().parse().unwrap());
    }
    let mut opaths:Vec<&str>=Vec::with_capacity(p);
    let mut okind:Vec<u8>=Vec::with_capacity(p);
    let mut osize:Vec<i64>=Vec::with_capacity(p);
    let mut odig:Vec<&str>=Vec::with_capacity(p);
    for _ in 0..p{
        opaths.push(it.next().unwrap());
        okind.push(it.next().unwrap().as_bytes()[0]);
        osize.push(it.next().unwrap().parse().unwrap());
        odig.push(it.next().unwrap());
    }
    if n==0{writeln!(w,"INVALID EMPTY").unwrap();return;}
    for i in 0..n{
        if !safe(paths[i]){writeln!(w,"INVALID PATH {}",i+1).unwrap();return;}
    }
    let mut seen:HashMap<&str,usize>=HashMap::new();
    for i in 0..n{
        if let Some(&f)=seen.get(paths[i]){
            writeln!(w,"INVALID DUP {} {}",f+1,i+1).unwrap();return;
        }
        seen.insert(paths[i],i);
    }
    {
        let mut idx:Vec<usize>=(0..n).collect();
        idx.sort_by(|&x,&y| paths[x].cmp(paths[y]));
        let mut best:Option<(&str,&str)>=None;
        for k in 0..n{
            let a=paths[idx[k]];
            for j in k+1..n{
                let bb=paths[idx[j]];
                if bb.len()>a.len() && bb.as_bytes()[a.len()]==b'/' && &bb[..a.len()]==a{
                    let cand=(a,bb);
                    match best{
                        None=>best=Some(cand),
                        Some(cur)=>{if cand<cur{best=Some(cand);}}
                    }
                }else{break;}
            }
        }
        if let Some((a,bb))=best{
            writeln!(w,"INVALID PREFIX {} {}",a,bb).unwrap();return;
        }
    }
    for i in 0..n{
        if sizes[i]<0{writeln!(w,"INVALID SIZE {}",i+1).unwrap();return;}
    }
    {
        let mut acc:i128=0;
        for i in 0..n{
            acc+=sizes[i] as i128;
            if acc>limit as i128{writeln!(w,"INVALID TOTAL {}",i+1).unwrap();return;}
        }
    }
    {
        let mut m:HashMap<&str,(usize,i64)>=HashMap::new();
        for i in 0..n{
            match m.get(digs[i]){
                Some(&(f,sz))=>{
                    if sz!=sizes[i]{writeln!(w,"INVALID DIGEST {} {}",f+1,i+1).unwrap();return;}
                },
                None=>{m.insert(digs[i],(i,sizes[i]));}
            }
        }
    }
    let mut obs:HashMap<&str,(u8,i64,&str)>=HashMap::new();
    for k in 0..p{
        if !safe(opaths[k]){continue;}
        obs.insert(opaths[k],(okind[k],osize[k],odig[k]));
    }
    let mut reusable=vec![false;n];
    for i in 0..n{
        let pth=paths[i];
        let mut ok=false;
        if let Some(&(k,sz,d))=obs.get(pth){
            if k==b'F' && sz==sizes[i] && d==digs[i]{ok=true;}
        }
        if ok{
            let bytes=pth.as_bytes();
            for j in 0..bytes.len(){
                if bytes[j]==b'/'{
                    let anc=&pth[..j];
                    if let Some(&(k,_,_))=obs.get(anc){
                        if k==b'F'||k==b'L'{ok=false;break;}
                    }
                }
            }
        }
        reusable[i]=ok;
    }
    let mut groups:HashMap<(&str,i64),Vec<usize>>=HashMap::new();
    for i in 0..n{
        groups.entry((digs[i],sizes[i])).or_insert_with(Vec::new).push(i);
    }
    let mut action=vec![0u8;n];
    let mut source=vec![0usize;n];
    for (_k,v) in groups.iter(){
        let mut first_reuse:Option<usize>=None;
        for &i in v.iter(){if reusable[i]{first_reuse=Some(i);break;}}
        match first_reuse{
            Some(r)=>{
                for &i in v.iter(){
                    if reusable[i]{action[i]=0;source[i]=i;}
                    else{action[i]=2;source[i]=r;}
                }
            },
            None=>{
                let f=v[0];
                for &i in v.iter(){
                    if i==f{action[i]=1;source[i]=i;}
                    else{action[i]=2;source[i]=f;}
                }
            }
        }
    }
    let mut fetches=0i64;
    let mut ops=0i64;
    for i in 0..n{
        if action[i]==1{fetches+=1;ops+=1;}
        else if action[i]==2{ops+=1;}
    }
    let mut waiting:Vec<Vec<usize>>=vec![Vec::new();n];
    let mut ready:BinaryHeap<Reverse<usize>>=BinaryHeap::new();
    for i in 0..n{
        if action[i]==1{ready.push(Reverse(i));}
        else if action[i]==2{
            let s=source[i];
            if action[s]==0{ready.push(Reverse(i));}
            else{waiting[s].push(i);}
        }
    }
    let mut freew:BinaryHeap<Reverse<i64>>=BinaryHeap::new();
    for x in 1..=b{freew.push(Reverse(x));}
    let mut running:BinaryHeap<Reverse<(i64,i64,usize)>>=BinaryHeap::new();
    let mut res:Vec<(i64,i64,usize,i64)>=Vec::with_capacity(ops as usize);
    let mut t:i64=0;
    loop{
        while let Some(&Reverse((ft,_,_)))=running.peek(){
            if ft==t{
                let Reverse((ft2,wk,idx))=running.pop().unwrap();
                freew.push(Reverse(wk));
                let _=ft2;
                for &c in waiting[idx].iter(){ready.push(Reverse(c));}
                waiting[idx].clear();
            }else{break;}
        }
        while !ready.is_empty() && !freew.is_empty(){
            let Reverse(idx)=ready.pop().unwrap();
            let Reverse(wk)=freew.pop().unwrap();
            let fin=t+durs[idx];
            running.push(Reverse((fin,wk,idx)));
            res.push((fin,wk,idx,t));
        }
        if running.is_empty(){break;}
        let Reverse((nt,_,_))=*running.peek().unwrap();
        t=nt;
    }
    res.sort();
    writeln!(w,"OK {} {} {}",n,fetches,ops).unwrap();
    for &(fin,wk,idx,st) in res.iter(){
        writeln!(w,"DONE {} {} {} {}",idx+1,wk,st,fin).unwrap();
    }
    for i in 0..n{
        let a=match action[i]{0=>"REUSE",1=>"FETCH",_=>"COPY"};
        writeln!(w,"FILE {} {} {} {} {} {}",i+1,paths[i],sizes[i],digs[i],a,source[i]+1).unwrap();
    }
}