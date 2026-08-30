use std::io::{self,Read,Write};
use std::collections::HashMap;

const INF: i64 = 1<<60;

struct Seg{mina:Vec<i64>,minb:Vec<i64>,la:Vec<i64>,ls:Vec<u8>,n:usize}

impl Seg{
    fn new(n:usize)->Seg{
        Seg{mina:vec![INF;4*n+4],minb:vec![INF;4*n+4],la:vec![0i64;4*n+4],ls:vec![0u8;4*n+4],n:n}
    }
    fn assign_node(&mut self,x:usize,t:u8){
        if t==1{
            let m=if self.mina[x]<self.minb[x]{self.mina[x]}else{self.minb[x]};
            self.mina[x]=m; self.minb[x]=INF;
        }else{
            let m=if self.mina[x]<self.minb[x]{self.mina[x]}else{self.minb[x]};
            self.minb[x]=m; self.mina[x]=INF;
        }
        self.ls[x]=t;
    }
    fn add_node(&mut self,x:usize,v:i64){
        if self.mina[x]<INF{self.mina[x]+=v;}
        if self.minb[x]<INF{self.minb[x]+=v;}
        self.la[x]+=v;
    }
    fn push(&mut self,x:usize){
        let t=self.ls[x];
        if t!=0{
            self.assign_node(2*x,t);
            self.assign_node(2*x+1,t);
            self.ls[x]=0;
        }
        let a=self.la[x];
        if a!=0{
            self.add_node(2*x,a);
            self.add_node(2*x+1,a);
            self.la[x]=0;
        }
    }
    fn pull(&mut self,x:usize){
        self.mina[x]=if self.mina[2*x]<self.mina[2*x+1]{self.mina[2*x]}else{self.mina[2*x+1]};
        self.minb[x]=if self.minb[2*x]<self.minb[2*x+1]{self.minb[2*x]}else{self.minb[2*x+1]};
    }
    fn range_add(&mut self,x:usize,lo:usize,hi:usize,l:usize,r:usize,v:i64){
        if r<lo||hi<l{return;}
        if l<=lo&&hi<=r{self.add_node(x,v);return;}
        self.push(x);
        let mid=(lo+hi)/2;
        self.range_add(2*x,lo,mid,l,r,v);
        self.range_add(2*x+1,mid+1,hi,l,r,v);
        self.pull(x);
    }
    fn range_assign(&mut self,x:usize,lo:usize,hi:usize,l:usize,r:usize,t:u8){
        if r<lo||hi<l{return;}
        if l<=lo&&hi<=r{self.assign_node(x,t);return;}
        self.push(x);
        let mid=(lo+hi)/2;
        self.range_assign(2*x,lo,mid,l,r,t);
        self.range_assign(2*x+1,mid+1,hi,l,r,t);
        self.pull(x);
    }
    fn create(&mut self,x:usize,lo:usize,hi:usize,pos:usize){
        if lo==hi{
            self.mina[x]=INF;
            self.minb[x]=0;
            self.la[x]=0;
            self.ls[x]=0;
            return;
        }
        self.push(x);
        let mid=(lo+hi)/2;
        if pos<=mid{self.create(2*x,lo,mid,pos);}else{self.create(2*x+1,mid+1,hi,pos);}
        self.pull(x);
    }
    fn del_leftmost(&mut self,x:usize,lo:usize,hi:usize)->usize{
        if lo==hi{
            self.mina[x]=INF;
            self.minb[x]=INF;
            self.la[x]=0;
            self.ls[x]=0;
            return lo;
        }
        self.push(x);
        let mid=(lo+hi)/2;
        let res;
        if self.mina[2*x]==0{
            res=self.del_leftmost(2*x,lo,mid);
        }else{
            res=self.del_leftmost(2*x+1,mid+1,hi);
        }
        self.pull(x);
        res
    }
    fn root_mina(&self)->i64{self.mina[1]}
    fn size(&self)->usize{self.n}
}

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let toks:Vec<&str>=s.split_whitespace().collect();
    if toks.is_empty(){return;}
    let q:usize=toks[0].parse().unwrap();
    let mut i=1usize;
    let mut cnt=0usize;
    for _ in 0..q{
        if i>=toks.len(){break;}
        let op=toks[i];
        if op=="C"{cnt+=1;i+=2;}
        else if op=="S"{i+=4;}
        else{i+=2;}
    }
    let n=if cnt==0{1}else{cnt};
    let mut seg=Seg::new(n);
    let mut names:Vec<&str>=Vec::with_capacity(cnt);
    let mut res:Vec<i64>=vec![-1;cnt+1];
    let mut map:HashMap<String,(usize,usize)>=HashMap::new();
    let mut created=0usize;
    i=1;
    let nn=seg.size();
    for ev in 1..=q{
        if i>=toks.len(){break;}
        let op=toks[i];
        if op=="C"{
            let name=toks[i+1];
            i+=2;
            created+=1;
            names.push(name);
            seg.create(1,1,nn,created);
        }else if op=="S"{
            let id=toks[i+1].to_string();
            let l:usize=toks[i+2].parse().unwrap();
            let r:usize=toks[i+3].parse().unwrap();
            i+=4;
            map.insert(id,(l,r));
            seg.range_add(1,1,nn,l,r,1);
        }else{
            let id=toks[i+1].to_string();
            i+=2;
            let (l,r)=*map.get(&id).unwrap();
            seg.range_add(1,1,nn,l,r,-1);
            if op=="O"{
                seg.range_assign(1,1,nn,l,r,1);
            }else{
                seg.range_assign(1,1,nn,l,r,2);
            }
        }
        while seg.root_mina()==0{
            let p=seg.del_leftmost(1,1,nn);
            res[p]=ev as i64;
        }
    }
    let out=io::stdout();
    let mut w=io::BufWriter::new(out.lock());
    let mut buf=String::new();
    buf.push_str(&created.to_string());
    for k in 1..=created{
        buf.push(' ');
        buf.push_str(names[k-1]);
        buf.push(' ');
        if res[k]<0{buf.push_str("ORPHAN");}else{buf.push_str(&res[k].to_string());}
    }
    buf.push('\n');
    w.write_all(buf.as_bytes()).unwrap();
}