use std::io::{self,Read,Write};
use std::collections::HashMap;

struct Panel{
    open:bool,
    prompt:usize,
    status:u8,
    detail_err:usize,
    detail_req:u64,
    cands:Vec<usize>,
    sel:Vec<bool>,
}

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it=s.split_ascii_whitespace();
    let mut strs:Vec<&str>=Vec::new();
    let mut interner:HashMap<&str,usize>=HashMap::new();
    let n:usize=it.next().unwrap().parse().unwrap();
    let q:usize=it.next().unwrap().parse().unwrap();
    let mut panels:Vec<Panel>=Vec::with_capacity(n+1);
    for _ in 0..n+1{
        panels.push(Panel{open:false,prompt:usize::MAX,status:0,detail_err:usize::MAX,detail_req:0,cands:Vec::new(),sel:Vec::new()});
    }
    let mut req_owner:HashMap<u64,usize>=HashMap::new();
    let mut pending:Vec<u64>=vec![0;n+1];
    let mut req_counter:u64=0;
    let mut selmap:HashMap<usize,(usize,usize)>=HashMap::new();

    macro_rules! intern{
        ($t:expr)=>{{
            let t:&str=$t;
            match interner.get(t){
                Some(&i)=>i,
                None=>{
                    let i=strs.len();
                    strs.push(t);
                    interner.insert(t,i);
                    i
                }
            }
        }};
    }

    fn clear_panel(p:&mut Panel,idx:usize,selmap:&mut HashMap<usize,(usize,usize)>,pending:&mut Vec<u64>){
        for (j,&c) in p.cands.iter().enumerate(){
            if p.sel[j]{
                if let Some(&(pi,pj))=selmap.get(&c){
                    if pi==idx && pj==j{ selmap.remove(&c); }
                }
            }
        }
        p.cands.clear();
        p.sel.clear();
        p.open=false;
        p.prompt=usize::MAX;
        p.status=0;
        p.detail_err=usize::MAX;
        p.detail_req=0;
        pending[idx]=0;
    }

    for _ in 0..q{
        let op=match it.next(){Some(x)=>x,None=>break};
        match op{
            "OPEN"=>{
                let i:usize=it.next().unwrap().parse().unwrap();
                let p=it.next().unwrap();
                let pi=intern!(p);
                clear_panel(&mut panels[i],i,&mut selmap,&mut pending);
                panels[i].open=true;
                panels[i].prompt=pi;
                panels[i].status=0;
            },
            "EDIT"=>{
                let i:usize=it.next().unwrap().parse().unwrap();
                let p=it.next().unwrap();
                let pi=intern!(p);
                if panels[i].open{ panels[i].prompt=pi; }
            },
            "START"=>{
                let i:usize=it.next().unwrap().parse().unwrap();
                req_counter+=1;
                let r=req_counter;
                if panels[i].open{
                    {
                        let p=&mut panels[i];
                        for (j,&c) in p.cands.iter().enumerate(){
                            if p.sel[j]{
                                if let Some(&(a,b))=selmap.get(&c){
                                    if a==i&&b==j{ selmap.remove(&c); }
                                }
                            }
                        }
                        p.cands.clear();
                        p.sel.clear();
                        p.detail_err=usize::MAX;
                        p.status=1;
                        p.detail_req=r;
                    }
                    pending[i]=r;
                    req_owner.insert(r,i);
                }
            },
            "SUCCESS"=>{
                let r:u64=it.next().unwrap().parse().unwrap();
                let k:usize=it.next().unwrap().parse().unwrap();
                let mut names:Vec<usize>=Vec::with_capacity(k);
                for _ in 0..k{
                    let t=it.next().unwrap();
                    names.push(intern!(t));
                }
                let owner=req_owner.get(&r).copied();
                if let Some(i)=owner{
                    if panels[i].open && pending[i]==r{
                        pending[i]=0;
                        let p=&mut panels[i];
                        p.status=2;
                        p.detail_err=usize::MAX;
                        p.sel=vec![false;names.len()];
                        p.cands=names;
                    }
                }
            },
            "VALIDATION"|"SERVICE"=>{
                let r:u64=it.next().unwrap().parse().unwrap();
                let e=it.next().unwrap();
                let ei=intern!(e);
                let owner=req_owner.get(&r).copied();
                if let Some(i)=owner{
                    if panels[i].open && pending[i]==r{
                        pending[i]=0;
                        {
                            let p=&mut panels[i];
                            for (j,&c) in p.cands.iter().enumerate(){
                                if p.sel[j]{
                                    if let Some(&(a,b))=selmap.get(&c){
                                        if a==i&&b==j{ selmap.remove(&c); }
                                    }
                                }
                            }
                            p.cands.clear();
                            p.sel.clear();
                            p.status=if op=="VALIDATION"{3}else{4};
                            p.detail_err=ei;
                        }
                    }
                }
            },
            "TOGGLE"=>{
                let i:usize=it.next().unwrap().parse().unwrap();
                let c=it.next().unwrap();
                let ci=intern!(c);
                if panels[i].open && panels[i].status==2{
                    let mut pos=usize::MAX;
                    for (j,&x) in panels[i].cands.iter().enumerate(){
                        if x==ci{ pos=j; break; }
                    }
                    if pos!=usize::MAX{
                        if panels[i].sel[pos]{
                            panels[i].sel[pos]=false;
                            if let Some(&(a,b))=selmap.get(&ci){
                                if a==i&&b==pos{ selmap.remove(&ci); }
                            }
                        }else{
                            if let Some(&(a,b))=selmap.get(&ci){
                                panels[a].sel[b]=false;
                            }
                            panels[i].sel[pos]=true;
                            selmap.insert(ci,(i,pos));
                        }
                    }
                }
            },
            "CLOSE"=>{
                let i:usize=it.next().unwrap().parse().unwrap();
                clear_panel(&mut panels[i],i,&mut selmap,&mut pending);
            },
            "RESET"=>{
                let l:usize=it.next().unwrap().parse().unwrap();
                let r:usize=it.next().unwrap().parse().unwrap();
                for i in l..=r{
                    clear_panel(&mut panels[i],i,&mut selmap,&mut pending);
                }
            },
            _=>{}
        }
    }

    let stdout=io::stdout();
    let mut out=io::BufWriter::new(stdout.lock());
    let mut buf=String::new();
    for i in 1..=n{
        let p=&panels[i];
        buf.push_str(&i.to_string());
        if !p.open{
            buf.push_str(" CLOSED - IDLE - 0\n");
        }else{
            buf.push_str(" OPEN ");
            buf.push_str(strs[p.prompt]);
            buf.push(' ');
            match p.status{
                0=>{buf.push_str("IDLE -");},
                1=>{buf.push_str("LOADING ");buf.push_str(&p.detail_req.to_string());},
                2=>{buf.push_str("READY -");},
                3=>{buf.push_str("VALIDATION ");buf.push_str(strs[p.detail_err]);},
                _=>{buf.push_str("SERVICE ");buf.push_str(strs[p.detail_err]);},
            }
            if p.status==2{
                buf.push(' ');
                buf.push_str(&p.cands.len().to_string());
                for (j,&c) in p.cands.iter().enumerate(){
                    buf.push(' ');
                    buf.push_str(strs[c]);
                    buf.push(' ');
                    buf.push(if p.sel[j]{'1'}else{'0'});
                }
            }else{
                buf.push_str(" 0");
            }
            buf.push('\n');
        }
        if buf.len()>1<<16{ out.write_all(buf.as_bytes()).unwrap(); buf.clear(); }
    }
    out.write_all(buf.as_bytes()).unwrap();
}