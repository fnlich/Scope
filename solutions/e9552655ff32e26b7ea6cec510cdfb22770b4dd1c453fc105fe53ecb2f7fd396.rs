use std::io::{self,Read,Write};

fn main(){
    let mut s=String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it=s.split_ascii_whitespace();
    let out=io::stdout();
    let mut o=io::BufWriter::new(out.lock());
    let f:i128=it.next().unwrap().parse().unwrap();
    let l:i128=it.next().unwrap().parse().unwrap();
    let r:i128=it.next().unwrap().parse().unwrap();
    let m:i128=it.next().unwrap().parse().unwrap();
    let w:i128=it.next().unwrap().parse().unwrap();
    let h:i128=it.next().unwrap().parse().unwrap();
    let q:usize=it.next().unwrap().parse().unwrap();
    let e:i128=if r<f {r} else {f};
    let mut p:i128=l;
    let mut b:i128=0;
    let mut d:i128=0;
    let mut peak:i128=0;
    let mut live=true;
    let mut attached=false;
    let mut enabled=false;
    let mut active=false;
    let mut req:i128=0;
    let mut queued=false;
    let mut reason=String::new();
    let mut buf=String::new();
    if l>=e {
        live=false;
        reason="EMPTY".to_string();
        buf.push_str("END 0 EMPTY\n");
    }
    for idx in 1..=q {
        let tok=match it.next(){Some(t)=>t,None=>break};
        match tok {
            "ATTACH"=>{
                if live && !attached {
                    attached=true;
                    enabled=true;
                    if live && attached && enabled && b<h && p<e {
                        if !active {
                            let a=p;
                            let bb=if p+m<e {p+m} else {e};
                            active=true; req=bb-a; queued=false;
                            buf.push_str(&format!("READ {} {} {}\n",idx,a,bb));
                        } else { queued=true; }
                    }
                }
            },
            "PAUSE"=>{
                if live && attached { enabled=false; queued=false; }
            },
            "RESUME"=>{
                if live && attached {
                    enabled=true;
                    if b<h && p<e {
                        if !active {
                            let a=p;
                            let bb=if p+m<e {p+m} else {e};
                            active=true; req=bb-a; queued=false;
                            buf.push_str(&format!("READ {} {} {}\n",idx,a,bb));
                        } else { queued=true; }
                    }
                }
            },
            "TAKE"=>{
                let x:i128=it.next().unwrap().parse().unwrap();
                let removed=if x<b {x} else {b};
                let oldb=b;
                b-=removed;
                let unmet=x-removed;
                let mut added=0i128;
                if live { d+=unmet; added=unmet; }
                if attached && ((oldb>w && b<=w) || added>0) {
                    enabled=true;
                    if live && b<h && p<e {
                        if !active {
                            let a=p;
                            let bb=if p+m<e {p+m} else {e};
                            active=true; req=bb-a; queued=false;
                            buf.push_str(&format!("READ {} {} {}\n",idx,a,bb));
                        } else { queued=true; }
                    }
                }
            },
            "DONE"=>{
                let x:i128=it.next().unwrap().parse().unwrap();
                if !active { continue; }
                queued=false;
                active=false;
                let rq=req;
                if x>0 {
                    p+=x;
                    let sat=if d<x {d} else {x};
                    d-=sat;
                    b+=x-sat;
                    if b>peak {peak=b;}
                }
                if x==0 {
                    live=false; queued=false; d=0; active=false;
                    reason="EOF".to_string();
                    buf.push_str(&format!("END {} EOF\n",idx));
                } else if x<rq {
                    live=false; queued=false; d=0; active=false;
                    reason="SHORT".to_string();
                    buf.push_str(&format!("END {} SHORT\n",idx));
                } else if p==e {
                    live=false; queued=false; d=0; active=false;
                    reason="RANGE".to_string();
                    buf.push_str(&format!("END {} RANGE\n",idx));
                } else {
                    if b>=h {
                        enabled=false; queued=false;
                    } else {
                        if live && attached && enabled && p<e {
                            let a=p;
                            let bb=if p+m<e {p+m} else {e};
                            active=true; req=bb-a; queued=false;
                            buf.push_str(&format!("READ {} {} {}\n",idx,a,bb));
                        }
                    }
                }
            },
            "FAIL"|"CANCEL"|"CLOSE"=>{
                if live {
                    live=false; active=false; queued=false; d=0;
                    reason=tok.to_string();
                    buf.push_str(&format!("END {} {}\n",idx,tok));
                }
            },
            _=>{}
        }
    }
    let state=if !live { format!("END_{}",reason) }
        else if active { "READING".to_string() }
        else if attached { "READY".to_string() }
        else { "UNATTACHED".to_string() };
    buf.push_str(&format!("FINAL {} {} {} {} {}\n",state,p,b,d,peak));
    o.write_all(buf.as_bytes()).unwrap();
}