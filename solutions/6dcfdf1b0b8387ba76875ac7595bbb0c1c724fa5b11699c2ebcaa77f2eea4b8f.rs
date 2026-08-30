use std::io::{self,Read,Write};

fn main(){
    let mut inp=String::new();
    io::stdin().read_to_string(&mut inp).unwrap();
    let mut it=inp.split_ascii_whitespace();
    let n:usize=it.next().unwrap().parse().unwrap();
    let q:usize=it.next().unwrap().parse().unwrap();
    let mut mk=vec![0u8;4*(n+1)];
    let mut ot=vec![0u8;4*(n+1)];
    let mut ids:Vec<usize>=Vec::new();
    let mut seen=[usize::MAX;16];
    let mut seq:Vec<u8>=Vec::new();
    for id in 1..=n{
        let t=it.next().unwrap();
        if t=="P"{
            let op=it.next().unwrap();
            let obs:u8=match op{
                "LOAD"=>1,
                "STORE"=>2,
                "BARRIER"=>32,
                "CALL"=>8,
                "NOP"=>0,
                "ANNUL"=>0,
                "JUMP"=>4|64,
                "DJUMP"=>4|64,
                "RETURN"=>16|64,
                "STOP"=>64,
                _=>4,
            };
            let nout:u8=match op{
                "LOAD"|"STORE"|"BARRIER"|"CALL"|"NOP"=>1,
                "ANNUL"=>2,
                "DJUMP"=>4,
                "CBR"=>1,
                _=>0,
            };
            for s in 0..4usize{
                let a=s&1;
                let d=s&2;
                let (m,o):(u8,u8)=if a==1{
                    if d!=0 {(0,0)} else {(0,1)}
                } else if d!=0{
                    (obs,0)
                } else {
                    (obs,nout)
                };
                mk[4*id+s]=m;
                ot[4*id+s]=o;
            }
        } else if t=="S"{
            let k:usize=it.next().unwrap().parse().unwrap();
            ids.clear();
            for _ in 0..k{
                let c:usize=it.next().unwrap().parse().unwrap();
                ids.push(c);
            }
            for s in 0..4usize{
                let mut r:u8=1u8<<s;
                let mut m:u8=0;
                for &c in ids.iter(){
                    if r==0 {break;}
                    let mut nr:u8=0;
                    for t2 in 0..4usize{
                        if r&(1u8<<t2)!=0{
                            m|=mk[4*c+t2];
                            nr|=ot[4*c+t2];
                        }
                    }
                    r=nr;
                }
                mk[4*id+s]=m;
                ot[4*id+s]=r;
            }
        } else {
            let c:u64=it.next().unwrap().parse().unwrap();
            let ch:usize=it.next().unwrap().parse().unwrap();
            for s in 0..4usize{
                let r0:u8=1u8<<s;
                let mut m:u8=0;
                for x in seen.iter_mut(){*x=usize::MAX;}
                seq.clear();
                seq.push(r0);
                seen[r0 as usize]=0;
                let mut i:u64=0;
                let fin:u8;
                loop{
                    if i==c{ fin=seq[i as usize]; break; }
                    let r=seq[i as usize];
                    let mut nr:u8=0;
                    for t2 in 0..4usize{
                        if r&(1u8<<t2)!=0{
                            m|=mk[4*ch+t2];
                            nr|=ot[4*ch+t2];
                        }
                    }
                    let j=seen[nr as usize];
                    if j!=usize::MAX{
                        let jj=j as u64;
                        let p=(i+1)-jj;
                        let rem=(c-jj)%p;
                        fin=seq[(jj+rem) as usize];
                        break;
                    }
                    seq.push(nr);
                    seen[nr as usize]=(i+1) as usize;
                    i+=1;
                }
                mk[4*id+s]=m;
                ot[4*id+s]=fin;
            }
        }
    }
    let so=io::stdout();
    let mut out=io::BufWriter::new(so.lock());
    let mut buf=String::new();
    for _ in 0..q{
        let r:usize=it.next().unwrap().parse().unwrap();
        let m=mk[4*r];
        for b in 0..7{
            if b>0 {buf.push(' ');}
            buf.push(if m&(1u8<<b)!=0 {'1'} else {'0'});
        }
        buf.push('\n');
    }
    out.write_all(buf.as_bytes()).unwrap();
}