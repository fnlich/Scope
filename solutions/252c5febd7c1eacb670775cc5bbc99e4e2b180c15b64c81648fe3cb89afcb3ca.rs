use std::io::{self,Read,Write};

fn nclos(e:usize,na:&[u8;2])->u8{
    let mut s=1u8<<e;
    loop{
        let mut t=s;
        for x in 0..2{ if (s>>x)&1==1 {t|=na[x];} }
        if t==s {break}
        s=t;
    }
    s
}
fn eclos(m:usize,na:&[u8;2],ca:&[u8;2])->u8{
    let mut s=1u8<<m;
    loop{
        let mut t=s;
        for x in 0..2{ if (s>>x)&1==1 {t|=na[x]|ca[x];} }
        if t==s {break}
        s=t;
    }
    s
}
fn oru(v:&mut Vec<Vec<u64>>,dst:usize,src:usize){
    if dst==src {return}
    if dst<src{
        let (l,r)=v.split_at_mut(src);
        let x=&mut l[dst];
        let y=&r[0];
        for k in 0..x.len(){ x[k]|=y[k]; }
    } else {
        let (l,r)=v.split_at_mut(dst);
        let x=&mut r[0];
        let y=&l[src];
        for k in 0..x.len(){ x[k]|=y[k]; }
    }
}

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it=s.split_ascii_whitespace();
    let out=io::stdout();
    let mut o=io::BufWriter::new(out.lock());
    let n:usize=match it.next(){Some(x)=>x.parse().unwrap_or(0),None=>0};
    let mut kind=vec![0u8;n+1];
    let mut c1=vec![0usize;n+1];
    let mut c2=vec![0usize;n+1];
    let mut posid=vec![0usize;n+1];
    let mut masks0:Vec<[u64;4]>=Vec::new();
    for i in 1..=n{
        let t=match it.next(){Some(x)=>x,None=>break};
        match t{
            "EMPTY"=>{kind[i]=0;}
            "FLIP"=>{kind[i]=1;}
            "CLASS"=>{
                kind[i]=2;
                let k:usize=it.next().unwrap_or("0").parse().unwrap_or(0);
                let mut m=[0u64;4];
                for _ in 0..k{
                    let l:usize=it.next().unwrap_or("0").parse().unwrap_or(0);
                    let r:usize=it.next().unwrap_or("0").parse().unwrap_or(0);
                    let mut b=l;
                    while b<=r && b<256 { m[b>>6]|=1u64<<(b&63); b+=1; }
                }
                masks0.push(m);
                posid[i]=masks0.len();
            }
            "ALT"=>{kind[i]=3;c1[i]=it.next().unwrap_or("0").parse().unwrap_or(0);c2[i]=it.next().unwrap_or("0").parse().unwrap_or(0);}
            "CAT"=>{kind[i]=4;c1[i]=it.next().unwrap_or("0").parse().unwrap_or(0);c2[i]=it.next().unwrap_or("0").parse().unwrap_or(0);}
            "STAR"=>{kind[i]=5;c1[i]=it.next().unwrap_or("0").parse().unwrap_or(0);}
            "PLUS"=>{kind[i]=6;c1[i]=it.next().unwrap_or("0").parse().unwrap_or(0);}
            "OPT"=>{kind[i]=7;c1[i]=it.next().unwrap_or("0").parse().unwrap_or(0);}
            _=>{kind[i]=0;}
        }
    }
    let p=masks0.len();
    if p==0 || n==0{
        writeln!(o,"YES").unwrap();
        return;
    }
    let mut masks1:Vec<[u64;4]>=Vec::with_capacity(p);
    for m in masks0.iter(){
        let mut q=*m;
        for b in 0..256usize{
            if (m[b>>6]>>(b&63))&1==1{
                let ob:i32= if b>=97 && b<=122 {b as i32-32} else if b>=65 && b<=90 {b as i32+32} else {-1};
                if ob>=0{
                    let ob=ob as usize;
                    q[ob>>6]|=1u64<<(ob&63);
                }
            }
        }
        masks1.push(q);
    }
    let ns=2*p;
    let w=(ns+63)/64;

    let mut nex=vec![[0u8;2];n+1];
    let mut cex=vec![[0u8;2];n+1];
    for i in 1..=n{
        match kind[i]{
            0=>{for m in 0..2{nex[i][m]=1u8<<m;}}
            1=>{for m in 0..2{nex[i][m]=1u8<<(1-m);}}
            2=>{for m in 0..2{cex[i][m]=1u8<<m;}}
            3=>{
                let (a,b)=(c1[i],c2[i]);
                for m in 0..2{
                    nex[i][m]=nex[a][m]|nex[b][m];
                    cex[i][m]=cex[a][m]|cex[b][m];
                }
            }
            4=>{
                let (a,b)=(c1[i],c2[i]);
                for m in 0..2{
                    let na=nex[a][m]; let ca=cex[a][m];
                    let mut nn=0u8; let mut cc=0u8;
                    for e in 0..2{
                        if (na>>e)&1==1{ nn|=nex[b][e]; cc|=cex[b][e]; }
                        if (ca>>e)&1==1{ cc|=nex[b][e]|cex[b][e]; }
                    }
                    nex[i][m]=nn; cex[i][m]=cc;
                }
            }
            5|6=>{
                let a=c1[i];
                let na=[nex[a][0],nex[a][1]];
                let ca=[cex[a][0],cex[a][1]];
                for m in 0..2{
                    let ee=eclos(m,&na,&ca);
                    if kind[i]==5{
                        nex[i][m]=nclos(m,&na);
                    } else {
                        let mut r=0u8;
                        for e in 0..2{ if (na[m]>>e)&1==1 { r|=nclos(e,&na); } }
                        nex[i][m]=r;
                    }
                    let mut r=0u8;
                    for e in 0..2{
                        if (ee>>e)&1==1{
                            for e1 in 0..2{
                                if (ca[e]>>e1)&1==1{ r|=nclos(e1,&na); }
                            }
                        }
                    }
                    cex[i][m]=r;
                }
            }
            7=>{
                let a=c1[i];
                for m in 0..2{
                    nex[i][m]=(1u8<<m)|nex[a][m];
                    cex[i][m]=cex[a][m];
                }
            }
            _=>{}
        }
    }

    let mut entry=vec![0u8;n+1];
    entry[n]=1;
    for i in (1..=n).rev(){
        let e=entry[i];
        if e==0 {continue}
        match kind[i]{
            3=>{
                let (a,b)=(c1[i],c2[i]);
                if a>=1&&a<=n {entry[a]|=e;}
                if b>=1&&b<=n {entry[b]|=e;}
            }
            4=>{
                let (a,b)=(c1[i],c2[i]);
                if a>=1&&a<=n {entry[a]|=e;}
                let mut s2=0u8;
                for m in 0..2{ if (e>>m)&1==1 { s2|=nex[a][m]|cex[a][m]; } }
                if b>=1&&b<=n {entry[b]|=s2;}
            }
            5|6=>{
                let a=c1[i];
                if a>=1&&a<=n{
                    let na=[nex[a][0],nex[a][1]];
                    let ca=[cex[a][0],cex[a][1]];
                    let mut s2=0u8;
                    for m in 0..2{ if (e>>m)&1==1 { s2|=eclos(m,&na,&ca); } }
                    entry[a]|=s2;
                }
            }
            7=>{
                let a=c1[i];
                if a>=1&&a<=n {entry[a]|=e;}
            }
            _=>{}
        }
    }

    let mut firsts:Vec<Vec<u64>>=vec![vec![0u64;w];2*(n+1)];
    let mut lasts:Vec<Vec<u64>>=vec![vec![0u64;w];4*(n+1)];
    let mut follow:Vec<Vec<u64>>=vec![vec![0u64;w];ns];
    let mut tmps=vec![0u64;w];
    let mut tmpd=vec![0u64;w];

    for i in 1..=n{
        for m in 0..2{
            if (entry[i]>>m)&1==0 {continue}
            match kind[i]{
                2=>{
                    let idx=(posid[i]-1)*2+m;
                    firsts[i*2+m][idx>>6]|=1u64<<(idx&63);
                    lasts[(i*2+m)*2+m][idx>>6]|=1u64<<(idx&63);
                }
                3=>{
                    let (a,b)=(c1[i],c2[i]);
                    oru(&mut firsts,i*2+m,a*2+m);
                    oru(&mut firsts,i*2+m,b*2+m);
                    for e in 0..2{
                        oru(&mut lasts,(i*2+m)*2+e,(a*2+m)*2+e);
                        oru(&mut lasts,(i*2+m)*2+e,(b*2+m)*2+e);
                    }
                }
                4=>{
                    let (a,b)=(c1[i],c2[i]);
                    oru(&mut firsts,i*2+m,a*2+m);
                    for e in 0..2{
                        if (nex[a][m]>>e)&1==1 { oru(&mut firsts,i*2+m,b*2+e); }
                    }
                    for e2 in 0..2{
                        let dst=(i*2+m)*2+e2;
                        let ex=nex[a][m]|cex[a][m];
                        for e in 0..2{
                            if (ex>>e)&1==1 { oru(&mut lasts,dst,(b*2+e)*2+e2); }
                        }
                        for e in 0..2{
                            if (nex[b][e]>>e2)&1==1 { oru(&mut lasts,dst,(a*2+m)*2+e); }
                        }
                    }
                    for e in 0..2{
                        let sidx=(a*2+m)*2+e;
                        let didx=b*2+e;
                        let mut anyd=false;
                        for k in 0..w{ if firsts[didx][k]!=0 {anyd=true;break} }
                        if !anyd {continue}
                        let mut anys=false;
                        for k in 0..w{ if lasts[sidx][k]!=0 {anys=true;break} }
                        if !anys {continue}
                        for k in 0..w{
                            let mut bits=lasts[sidx][k];
                            while bits!=0{
                                let t=bits.trailing_zeros() as usize;
                                bits&=bits-1;
                                let st=k*64+t;
                                for kk in 0..w{
                                    follow[st][kk]|=firsts[didx][kk];
                                }
                            }
                        }
                    }
                }
                5|6=>{
                    let a=c1[i];
                    let na=[nex[a][0],nex[a][1]];
                    let ca=[cex[a][0],cex[a][1]];
                    let ee=eclos(m,&na,&ca);
                    let ncm=nclos(m,&na);
                    for e in 0..2{
                        if (ncm>>e)&1==1 { oru(&mut firsts,i*2+m,a*2+e); }
                    }
                    for e2 in 0..2{
                        let dst=(i*2+m)*2+e2;
                        for e in 0..2{
                            if (ee>>e)&1==0 {continue}
                            for e1 in 0..2{
                                if (nclos(e1,&na)>>e2)&1==1 {
                                    oru(&mut lasts,dst,(a*2+e)*2+e1);
                                }
                            }
                        }
                    }
                    for e1 in 0..2{
                        for k in 0..w{ tmps[k]=0; tmpd[k]=0; }
                        let