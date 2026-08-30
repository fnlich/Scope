use std::io::{self,Read,Write};

fn main(){
    let mut inp=Vec::new();
    io::stdin().read_to_end(&mut inp).unwrap();
    let mut pos=0usize;
    let data=&inp[..];
    let mut next_tok=|pos:&mut usize|->Option<(usize,usize)>{
        while *pos<data.len() && (data[*pos]==b' '||data[*pos]==b'\n'||data[*pos]==b'\r'||data[*pos]==b'\t'||data[*pos]==0x0b||data[*pos]==0x0c){*pos+=1;}
        if *pos>=data.len(){return None;}
        let st=*pos;
        while *pos<data.len() && !(data[*pos]==b' '||data[*pos]==b'\n'||data[*pos]==b'\r'||data[*pos]==b'\t'||data[*pos]==0x0b||data[*pos]==0x0c){*pos+=1;}
        Some((st,*pos))
    };
    let mut nextu=|pos:&mut usize|->u64{
        let (a,b)=next_tok(pos).unwrap();
        let mut v:u64=0;
        for i in a..b{ v=v*10+(data[i]-b'0') as u64;}
        v
    };

    let l=nextu(&mut pos) as usize;
    let p=nextu(&mut pos) as usize;
    let mut head=vec![-1i32;l+1];
    let mut nxt=vec![-1i32;p];
    let mut to=vec![0u32;p];
    let mut selfloop=vec![false;l+1];
    for i in 0..p{
        let a=nextu(&mut pos) as usize;
        let b=nextu(&mut pos) as usize;
        if a==b {selfloop[a]=true;}
        to[i]=b as u32;
        nxt[i]=head[a];
        head[a]=i as i32;
    }

    let n=l+1;
    let mut index=vec![0u32;n];
    let mut low=vec![0u32;n];
    let mut onstk=vec![false;n];
    let mut comp=vec![u32::MAX;n];
    let mut idx:u32=1;
    let mut stk:Vec<usize>=Vec::new();
    let mut ncomp:u32=0;
    let mut callstack:Vec<(usize,i32)>=Vec::new();
    for s in 1..=l{
        if index[s]!=0 {continue;}
        callstack.push((s,head[s]));
        index[s]=idx; low[s]=idx; idx+=1; stk.push(s); onstk[s]=true;
        while let Some(&mut(v,ref mut e))=callstack.last_mut(){
            if *e!=-1{
                let ei=*e as usize;
                let w=to[ei] as usize;
                *e=nxt[ei];
                if index[w]==0{
                    index[w]=idx; low[w]=idx; idx+=1; stk.push(w); onstk[w]=true;
                    callstack.push((w,head[w]));
                } else if onstk[w]{
                    if index[w]<low[v]{low[v]=index[w];}
                }
            } else {
                let v=v;
                callstack.pop();
                if low[v]==index[v]{
                    loop{
                        let w=stk.pop().unwrap();
                        onstk[w]=false;
                        comp[w]=ncomp;
                        if w==v{break;}
                    }
                    ncomp+=1;
                }
                if let Some(&mut(u,_))=callstack.last_mut(){
                    if low[v]<low[u]{low[u]=low[v];}
                }
            }
        }
    }

    let nc=ncomp as usize;
    let mut csize=vec![0u32;nc.max(1)];
    for v in 1..=l{ csize[comp[v] as usize]+=1;}
    let mut ccyc=vec![false;nc.max(1)];
    for v in 1..=l{
        let c=comp[v] as usize;
        if csize[c]>1 || selfloop[v]{ ccyc[c]=true;}
    }

    let words=(nc+63)/64;
    let mut clo=vec![0u64;words*nc.max(1)];
    let mut cedges:Vec<Vec<u32>>=vec![Vec::new();nc.max(1)];
    for v in 1..=l{
        let cv=comp[v] as usize;
        let mut e=head[v];
        while e!=-1{
            let w=to[e as usize] as usize;
            let cw=comp[w] as usize;
            if cw!=cv{ cedges[cv].push(cw as u32);}
            e=nxt[e as usize];
        }
    }
    for c in 0..nc{
        let base=c*words;
        clo[base+c/64]|=1u64<<(c%64);
        let list=std::mem::take(&mut cedges[c]);
        for &d in list.iter(){
            let db=(d as usize)*words;
            for k in 0..words{
                clo[base+k]|=clo[db+k];
            }
        }
        cedges[c]=list;
    }
    let reach=|a:usize,b:usize,clo:&Vec<u64>|->bool{
        let ca=comp[a] as usize; let cb=comp[b] as usize;
        (clo[ca*words+cb/64]>>(cb%64))&1==1
    };

    let r=nextu(&mut pos) as usize;
    use std::collections::HashMap;
    let mut gmap:HashMap<(u64,u64),usize>=HashMap::new();
    let mut gspan:Vec<(u64,u64)>=Vec::new();
    let mut grecs:Vec<Vec<(u64,u8,u32)>>=Vec::new();
    for _ in 0..r{
        let s=nextu(&mut pos);
        let e=nextu(&mut pos);
        let x=nextu(&mut pos);
        let (ka,_kb)=next_tok(&mut pos).unwrap();
        let k=data[ka];
        let c=nextu(&mut pos) as u32;
        let gi=match gmap.get(&(s,e)){
            Some(&i)=>i,
            None=>{
                let i=gspan.len();
                gmap.insert((s,e),i);
                gspan.push((s,e));
                grecs.push(Vec::new());
                i
            }
        };
        grecs[gi].push((x,k,c));
    }

    let ng=gspan.len();
    let mut results:Vec<String>=Vec::with_capacity(ng);
    let mut seenlab=vec![0u32;l+1];
    let mut stamp:u32=0;
    for gi in 0..ng{
        let recs=&grecs[gi];
        let mut bsyms:std::collections::HashSet<u64>=std::collections::HashSet::new();
        for &(x,k,_) in recs.iter(){ if k==b'B'{ bsyms.insert(x);} }
        stamp+=1;
        let mut labels:Vec<u32>=Vec::new();
        for &(x,k,c) in recs.iter(){
            if k==b'O' && bsyms.contains(&x){continue;}
            let ci=c as usize;
            if seenlab[ci]==stamp{continue;}
            seenlab[ci]=stamp;
            labels.push(c);
        }
        let mut cyc=false;
        for &c in labels.iter(){
            if ccyc[comp[c as usize] as usize]{cyc=true;break;}
        }
        if cyc{ results.push("CYCLE".to_string()); continue;}
        if labels.is_empty(){ results.push("CONFLICT".to_string()); continue;}
        let mut cand=labels[0];
        for i in 1..labels.len(){
            if !reach(cand as usize,labels[i] as usize,&clo){ cand=labels[i];}
        }
        let mut ok=true;
        for &c in labels.iter(){
            if !reach(cand as usize,c as usize,&clo){ok=false;break;}
        }
        if ok{ results.push(cand.to_string()); } else { results.push("CONFLICT".to_string()); }
    }

    let mut bys:Vec<(u64,u32)>=(0..ng).map(|i|(gspan[i].0,i as u32)).collect();
    bys.sort();
    let mut bye:Vec<(u64,u32)>=(0..ng).map(|i|(gspan[i].1,i as u32)).collect();
    bye.sort();

    let q=nextu(&mut pos) as usize;
    let out=io::stdout();
    let mut w=io::BufWriter::new(out.lock());
    let mut buf:Vec<u32>=Vec::new();
    for _ in 0..q{
        let ql=nextu(&mut pos);
        let qr=nextu(&mut pos);
        buf.clear();
        let lo=bys.partition_point(|&(v,_)| v<ql);
        let mut i=lo;
        while i<bys.len() && bys[i].0<=qr{ buf.push(bys[i].1); i+=1;}
        let lo2=bye.partition_point(|&(v,_)| v<ql);
        let mut j=lo2;
        while j<bye.len() && bye[j].0<=qr{ buf.push(bye[j].1); j+=1;}
        buf.sort_unstable();
        buf.dedup();
        w.write_all(buf.len().to_string().as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();
        for &gi in buf.iter(){
            let g=gi as usize;
            w.write_all(gspan[g].0.to_string().as_bytes()).unwrap();
            w.write_all(b" ").unwrap();
            w.write_all(gspan[g].1.to_string().as_bytes()).unwrap();
            w.write_all(b" ").unwrap();
            w.write_all(results[g].as_bytes()).unwrap();
            w.write_all(b"\n").unwrap();
        }
    }
}