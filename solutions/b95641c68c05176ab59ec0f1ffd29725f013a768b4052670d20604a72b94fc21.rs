use std::io::{self,Read,Write};

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it=s.split_ascii_whitespace();
    let mut next=||->i64{ it.next().unwrap().parse::<i64>().unwrap() };
    let vw=next(); let vh=next(); let l=next(); let r=next(); let t=next(); let d=next();
    let w0=next(); let h0=next();
    let w1=next(); let h1=next();
    let n=next() as usize;
    let mut pal0:Vec<u32>=Vec::with_capacity(n);
    for _ in 0..n { pal0.push(next() as u32); }
    let mut pal1:Vec<u32>=Vec::with_capacity(n);
    for _ in 0..n { pal1.push(next() as u32); }
    let c0=(w0*h0) as usize;
    let mut img0:Vec<u32>=Vec::with_capacity(c0);
    for _ in 0..c0 { img0.push(next() as u32); }
    let c1=(w1*h1) as usize;
    let mut img1:Vec<u32>=Vec::with_capacity(c1);
    for _ in 0..c1 { img1.push(next() as u32); }
    let q=next() as usize;
    struct Cmd{r:i64,op:u8,a:i64,b:i64,c:i64,e:i64}
    let mut cmds:Vec<Cmd>=Vec::with_capacity(q);
    for _ in 0..q {
        let rr=it.next().unwrap().parse::<i64>().unwrap();
        let op=it.next().unwrap().as_bytes()[0];
        match op {
            b'P'=>{let a=it.next().unwrap().parse::<i64>().unwrap();let b=it.next().unwrap().parse::<i64>().unwrap();let c=it.next().unwrap().parse::<i64>().unwrap();cmds.push(Cmd{r:rr,op,a,b,c,e:0});}
            b'R'=>{let a=it.next().unwrap().parse::<i64>().unwrap();let b=it.next().unwrap().parse::<i64>().unwrap();let c=it.next().unwrap().parse::<i64>().unwrap();let e=it.next().unwrap().parse::<i64>().unwrap();cmds.push(Cmd{r:rr,op,a,b,c,e});}
            b'O'|b'A'=>{let a=it.next().unwrap().parse::<i64>().unwrap();let b=it.next().unwrap().parse::<i64>().unwrap();let c=it.next().unwrap().parse::<i64>().unwrap();cmds.push(Cmd{r:rr,op,a,b,c,e:0});}
            _=>{let a=it.next().unwrap().parse::<i64>().unwrap();cmds.push(Cmd{r:rr,op,a,b:0,c:0,e:0});}
        }
    }
    let fw=l+vw+r;
    let fh=t+vh+d;
    let mut offx=[0i64;2];
    let mut offy=[0i64;2];
    let mut trans:usize=0;
    let mut border:usize=0;
    let has1 = w1>0 && h1>0;
    let out=io::stdout();
    let mut w=io::BufWriter::with_capacity(1<<20,out.lock());
    let mut buf:Vec<u8>=Vec::with_capacity((fw as usize)*9+2);
    let mut ci=0usize;
    let mut tmp:Vec<u32>=Vec::new();
    for py in 0..fh {
        while ci<cmds.len() && cmds[ci].r==py {
            let c=&cmds[ci];
            match c.op {
                b'P'=>{ let p= if c.a==0 {&mut pal0} else {&mut pal1}; p[c.b as usize]=c.c as u32; }
                b'R'=>{
                    let p= if c.a==0 {&mut pal0} else {&mut pal1};
                    let lo=c.b as usize; let hi=c.c as usize; let len=hi-lo+1;
                    let mut k=c.e % (len as i64);
                    if k<0 { k+=len as i64; }
                    let k=k as usize;
                    if k!=0 {
                        tmp.clear();
                        tmp.extend_from_slice(&p[lo..=hi]);
                        for i in 0..len {
                            p[lo+(i+k)%len]=tmp[i];
                        }
                    }
                }
                b'O'=>{ offx[c.a as usize]=c.b; offy[c.a as usize]=c.c; }
                b'A'=>{ let i=c.a as usize; offx[i]+=c.b; offy[i]+=c.c; }
                b'T'=>{ trans=c.a as usize; }
                _=>{ border=c.a as usize; }
            }
            ci+=1;
        }
        buf.clear();
        let bc=pal0[border];
        if py<t || py>=t+vh {
            for _ in 0..fw { push(&mut buf,bc); }
        } else {
            let y=py-t;
            let sy0=(((y+offy[0])%h0)+h0)%h0;
            let base0=(sy0*w0) as usize;
            let (base1,sy1w)= if has1 {
                let sy1=(((y+offy[1])%h1)+h1)%h1;
                ((sy1*w1) as usize,true)
            } else {(0,false)};
            for px in 0..fw {
                if px<l || px>=l+vw {
                    push(&mut buf,bc);
                } else {
                    let x=px-l;
                    let mut col:u32;
                    let sx0=(((x+offx[0])%w0)+w0)%w0;
                    col=pal0[img0[base0+sx0 as usize] as usize];
                    if sy1w {
                        let sx1=(((x+offx[1])%w1)+w1)%w1;
                        let idx=img1[base1+sx1 as usize] as usize;
                        if idx!=trans { col=pal1[idx]; }
                    }
                    push(&mut buf,col);
                }
            }
            buf.pop();
        }
        if py<t || py>=t+vh { if fw>0 { buf.pop(); } }
        buf.push(b'\n');
        w.write_all(&buf).unwrap();
    }
    w.flush().unwrap();
}

fn push(buf:&mut Vec<u8>,v:u32){
    let mut t=[0u8;10];
    let mut i=10;
    let mut x=v;
    if x==0 { i-=1; t[i]=b'0'; }
    while x>0 { i-=1; t[i]=b'0'+(x%10) as u8; x/=10; }
    buf.extend_from_slice(&t[i..]);
    buf.push(b' ');
}