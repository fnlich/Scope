use std::io::{self,Read,Write};

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it=s.split_ascii_whitespace();
    let n:i64=it.next().unwrap().parse().unwrap();
    let tri=it.next().unwrap().to_string();
    let ori=it.next().unwrap().to_string();
    let r:u64=it.next().unwrap().parse().unwrap();
    let out=io::stdout();
    let mut w=io::BufWriter::new(out.lock());
    if n==0{
        writeln!(w,"0").unwrap();
        return;
    }
    let t=(n*(n+1)/2) as usize;
    let nn=n as usize;
    let rows:usize = if n%2==0 {(n+1) as usize} else {n as usize};
    let upper = tri=="U";
    let conj_n = ori=="N";

    let mut p=vec![0usize;t];
    let mut b=vec![0u8;t];

    let par=((n-1)%2) as i64;
    let mut q:usize=0;

    let mut set=|d:i64,tt:i64,folded:bool,q:usize,p:&mut Vec<usize>,b:&mut Vec<u8>|{
        let (i,j)= if upper {(tt,tt+d)} else {(tt+d,tt)};
        let row=q%rows;
        let col=q/rows;
        let cols= if n%2==0 {(n/2) as usize} else {((n+1)/2) as usize};
        let src=row*cols+col;
        let dst= if upper {
            let jj=j as usize;
            (jj*(jj+1)/2)+(i as usize)
        } else {
            let jj=j as usize;
            let ii=i as usize;
            let base=jj*nn - (jj*(jj.wrapping_sub(1)))/2;
            base + (ii-jj)
        };
        p[src]=dst;
        let bit = if conj_n { folded } else { !folded };
        b[src]= if bit {1} else {0};
    };

    let mut d=if par==0 {0} else {1};
    while d<n {
        for tt in 0..(n-d){
            set(d,tt,false,q,&mut p,&mut b);
            q+=1;
        }
        d+=2;
    }
    let mut d2=n-1;
    while d2>=0 {
        if (d2%2)!=par {
            let mut tt=n-d2-1;
            while tt>=0 {
                set(d2,tt,true,q,&mut p,&mut b);
                q+=1;
                tt-=1;
            }
        }
        d2-=1;
    }

    let mut qf=vec![0usize;t];
    let mut bf=vec![0u8;t];
    let mut seen=vec![false;t];
    for st in 0..t{
        if seen[st]{continue;}
        let mut cyc:Vec<usize>=Vec::new();
        let mut bits:Vec<u8>=Vec::new();
        let mut x=st;
        loop{
            seen[x]=true;
            cyc.push(x);
            bits.push(b[x]);
            x=p[x];
            if x==st{break;}
        }
        let l=cyc.len();
        let mut pre=vec![0u8;l+1];
        for i in 0..l{pre[i+1]=pre[i]^bits[i];}
        let full=pre[l];
        let shift=(r%(l as u64)) as usize;
        let full_rounds=r/(l as u64);
        let fr=((full_rounds%2) as u8)*full;
        for i in 0..l{
            let jj=(i+shift)%l;
            qf[cyc[i]]=cyc[jj];
            let partial = if i+shift < l { pre[i+shift]^pre[i] } else { (pre[l]^pre[i])^pre[jj] };
            bf[cyc[i]]=fr^partial;
        }
    }

    let mut vis=vec![false;t];
    let mut cycles:Vec<Vec<usize>>=Vec::new();
    for st in 0..t{
        if vis[st]{continue;}
        let mut c=Vec::new();
        let mut x=st;
        loop{
            vis[x]=true;
            c.push(x);
            x=qf[x];
            if x==st{break;}
        }
        cycles.push(c);
    }
    let mut res=String::new();
    res.push_str(&cycles.len().to_string());
    res.push('\n');
    for c in &cycles{
        res.push_str(&c.len().to_string());
        for &x in c{
            res.push(' ');
            res.push_str(&x.to_string());
            res.push(' ');
            res.push_str(&bf[x].to_string());
        }
        res.push('\n');
    }
    w.write_all(res.as_bytes()).unwrap();
}