use std::io::{self,Read,Write};

#[derive(Clone,Copy)]
struct F{a:i128,lo:i128,hi:i128}

const INF:i128=1_000_000_000_000_000_000_000_000_000_000i128;

fn idf()->F{F{a:0,lo:-INF,hi:INF}}

fn clampv(x:i128,lo:i128,hi:i128)->i128{ if x<lo {lo} else if x>hi {hi} else {x} }

fn comb(f:F,g:F)->F{
    F{a:f.a+g.a, lo:clampv(f.lo+g.a,g.lo,g.hi), hi:clampv(f.hi+g.a,g.lo,g.hi)}
}

fn apply(f:F,x:i128)->i128{ clampv(x+f.a,f.lo,f.hi) }

fn ceil_div(a:i128,b:i128)->i128{ (a + b -1)/b }

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it=s.split_ascii_whitespace();
    let d:i128=it.next().unwrap().parse().unwrap();
    let w:i128=it.next().unwrap().parse().unwrap();
    let h:i128=it.next().unwrap().parse().unwrap();
    let pw:i128=it.next().unwrap().parse().unwrap();
    let ph:i128=it.next().unwrap().parse().unwrap();
    let x0:i128=it.next().unwrap().parse().unwrap();
    let y0:i128=it.next().unwrap().parse().unwrap();
    let x1:i128=it.next().unwrap().parse().unwrap();
    let y1:i128=it.next().unwrap().parse().unwrap();
    let n:usize=it.next().unwrap().parse().unwrap();
    let m:usize=it.next().unwrap().parse().unwrap();

    let sx=x0+x1;
    let sy=y0+y1;
    let w0=x1-x0;
    let h0=y1-y0;

    let mut gw=ceil_div(d*pw,w);
    if ((gw%2)+2)%2 != ((w0%2)+2)%2 { gw+=1; }
    let minw= if gw<w0 {gw} else {w0};
    let mut gh=ceil_div(d*ph,h);
    if ((gh%2)+2)%2 != ((h0%2)+2)%2 { gh+=1; }
    let minh= if gh<h0 {gh} else {h0};

    let maxw= if sx < 2*d-sx {sx} else {2*d-sx};
    let maxh= if sy < 2*d-sy {sy} else {2*d-sy};

    let mut sz=1usize;
    while sz<n {sz*=2;}
    let mut tw=vec![idf();2*sz];
    let mut th=vec![idf();2*sz];

    let mut readop=|it:&mut std::str::SplitAsciiWhitespace|->(F,F){
        let hd=it.next().unwrap();
        let dx:i128=it.next().unwrap().parse().unwrap();
        let dy:i128=it.next().unwrap().parse().unwrap();
        let mut fw=idf();
        let mut fh=idf();
        let bl=hd.contains('L');
        let br=hd.contains('R');
        let bt=hd.contains('T');
        let bb=hd.contains('B');
        if bl {fw=F{a:-2*dx,lo:minw,hi:maxw};}
        else if br {fw=F{a:2*dx,lo:minw,hi:maxw};}
        if bt {fh=F{a:-2*dy,lo:minh,hi:maxh};}
        else if bb {fh=F{a:2*dy,lo:minh,hi:maxh};}
        (fw,fh)
    };

    for i in 0..n{
        let (fw,fh)=readop(&mut it);
        tw[sz+i]=fw; th[sz+i]=fh;
    }
    for i in (1..sz).rev(){
        tw[i]=comb(tw[2*i],tw[2*i+1]);
        th[i]=comb(th[2*i],th[2*i+1]);
    }

    let out=io::stdout();
    let mut o=io::BufWriter::new(out.lock());
    let mut buf=String::new();

    for _ in 0..m{
        let c=it.next().unwrap();
        if c=="U"{
            let i:usize=it.next().unwrap().parse::<usize>().unwrap()-1;
            let (fw,fh)=readop(&mut it);
            let mut p=sz+i;
            tw[p]=fw; th[p]=fh;
            p/=2;
            while p>=1{
                tw[p]=comb(tw[2*p],tw[2*p+1]);
                th[p]=comb(th[2*p],th[2*p+1]);
                if p==1 {break;}
                p/=2;
            }
        } else {
            let l:usize=it.next().unwrap().parse::<usize>().unwrap()-1;
            let r:usize=it.next().unwrap().parse::<usize>().unwrap()-1;
            let mut lo=l+sz;
            let mut hi=r+sz+1;
            let mut lfw=idf(); let mut lfh=idf();
            let mut rfw=idf(); let mut rfh=idf();
            while lo<hi{
                if lo&1==1{
                    lfw=comb(lfw,tw[lo]); lfh=comb(lfh,th[lo]);
                    lo+=1;
                }
                if hi&1==1{
                    hi-=1;
                    rfw=comb(tw[hi],rfw); rfh=comb(th[hi],rfh);
                }
                lo/=2; hi/=2;
            }
            let fw=comb(lfw,rfw);
            let fh=comb(lfh,rfh);
            let lw=apply(fw,w0);
            let lh=apply(fh,h0);
            let a=(sx-lw)/2;
            let cc=(sx+lw)/2;
            let b=(sy-lh)/2;
            let dd=(sy+lh)/2;
            let ra=a*w/d;
            let rb=b*h/d;
            let rc=ceil_div(cc*w,d);
            let re=ceil_div(dd*h,d);
            buf.clear();
            buf.push_str(&a.to_string()); buf.push(' ');
            buf.push_str(&b.to_string()); buf.push(' ');
            buf.push_str(&cc.to_string()); buf.push(' ');
            buf.push_str(&dd.to_string()); buf.push(' ');
            buf.push_str(&ra.to_string()); buf.push(' ');
            buf.push_str(&rb.to_string()); buf.push(' ');
            buf.push_str(&rc.to_string()); buf.push(' ');
            buf.push_str(&re.to_string()); buf.push('\n');
            o.write_all(buf.as_bytes()).unwrap();
        }
    }
}