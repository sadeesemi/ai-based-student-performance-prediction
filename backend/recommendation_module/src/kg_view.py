"""Writes the standalone interactive knowledge-graph window (no CDN, works offline).

The page ships the whole catalogue side of the graph plus a compact index of every
learner, so the ego-network of any of the 5,000 students can be rebuilt live in the
browser (force-directed, draggable, zoomable, searchable).
"""
import json

from . import config

KINDS = ["student", "lesson", "topic", "resource", "tag", "risk", "style"]


def build_payload(profiles, resources, graph, recommendations, sample_students):
    nodes, index = [], {}

    def add(nid, kind, label, extra=None):
        if nid in index:
            return index[nid]
        index[nid] = len(nodes)
        nodes.append([nid, KINDS.index(kind), label, extra or {}])
        return index[nid]

    for _, r in resources.iterrows():
        add(f"L::{r['Lesson_Name']}", "lesson", r["Lesson_Name"])
    for _, r in resources.iterrows():
        add(f"T::{r['Topic']}", "topic", r["Topic"])
    for _, r in resources.iterrows():
        add(r["Resource_ID"], "resource", r["Resource_Title"],
            {"type": r["Resource_Type"], "level": r["Difficulty_Level"],
             "min": int(r["Estimated_Duration_Min"]), "lesson": r["Lesson_Name"],
             "topic": r["Topic"], "url": r["LMS_URL"]})
    for _, r in resources.iterrows():
        for t in r["tag_list"]:
            add(f"K::{t}", "tag", t)
    for lvl in ["Low Risk", "Medium Risk", "High Risk"]:
        add(f"R::{lvl}", "risk", lvl)
    for st in config.STYLE_TYPE_FIT:
        add(f"S::{st}", "style", st)

    links = []
    for u, v, d in graph.edges(data=True):
        if u in index and v in index:
            links.append([index[u], index[v], d.get("rel", "RELATED_TO")])

    students = {}
    for _, s in profiles.iterrows():
        sid = s["student_id"]
        gaps = [index[f"L::{g}"] for g in s["gap_lessons"] if f"L::{g}" in index]
        recs = [index[r["resource_id"]] for r in recommendations.get(sid, [])[:6] if r["resource_id"] in index]
        students[sid] = [s["Name"], s["Predicted_Risk_Level"], s["Learning_Style"], gaps, recs,
                         round(float(s["engagement_score"]), 1), round(float(s["performance_score"]), 1),
                         s["program"]]
    return {"nodes": nodes, "links": links, "students": students,
            "sample": list(sample_students), "kinds": KINDS}


def write(payload, stats, path=None):
    path = path or (config.OUTPUT_DIR / "knowledge_graph.html")
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__STATS__", json.dumps(stats, separators=(",", ":")))
    path.write_text(html, encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Module 03 - Live Knowledge Graph</title>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden;
font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0d2027;color:#e8eef0}
#app{display:flex;height:100%}
#side{width:296px;flex:0 0 296px;background:#0a1a20;border-right:1px solid #1d3a44;padding:16px;overflow-y:auto}
h1{font-size:15px;margin:0 0 2px}h1 span{color:#4fd1c5}
.sub{font-size:11px;color:#7f9aa3;margin-bottom:14px;line-height:1.5}
label.f{display:block;font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:#7f9aa3;margin:14px 0 5px}
input,select,button{font-family:inherit;font-size:12.5px}
input[type=text],select{width:100%;padding:8px 10px;border-radius:6px;border:1px solid #23414c;background:#0d2731;color:#e8eef0}
button{width:100%;margin-top:7px;padding:8px 10px;border-radius:6px;border:0;background:#0f6f7a;color:#fff;font-weight:600;cursor:pointer}
button.ghost{background:#16303a;color:#bcd3d9}
.row{display:flex;gap:6px}.row button{margin-top:7px}
.chk{display:flex;align-items:center;gap:7px;font-size:12px;margin:5px 0;cursor:pointer}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.stat{display:flex;justify-content:space-between;font-size:11.5px;padding:3px 0;color:#a9c2c9}
.stat b{font-family:ui-monospace,Menlo,monospace;color:#fff}
#info{margin-top:12px;background:#0d2731;border:1px solid #1d3a44;border-radius:8px;padding:10px;font-size:11.5px;line-height:1.55;display:none}
#info h4{margin:0 0 5px;font-size:12.5px}#info a{color:#4fd1c5}
#canvasWrap{flex:1;position:relative}canvas{display:block;cursor:grab}
#tip{position:absolute;pointer-events:none;background:#06141a;border:1px solid #24444f;border-radius:6px;
padding:7px 9px;font-size:11.5px;max-width:260px;display:none;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,.45)}
#hint{position:absolute;left:14px;bottom:12px;font-size:11px;color:#6d8b94}
#mode{position:absolute;left:14px;top:12px;font-size:12px;color:#9dbcc4;background:rgba(6,20,26,.75);
border:1px solid #1d3a44;border-radius:6px;padding:6px 10px}
.pill{display:inline-block;font-size:10px;padding:2px 7px;border-radius:20px;margin:2px 3px 0 0;background:#16303a;color:#bcd3d9}
.err{color:#ff9b8a;font-size:11.5px;margin-top:6px;display:none}
</style></head><body>
<div id="app">
<div id="side">
  <h1>Knowledge <span>Graph</span></h1>
  <div class="sub">Live student &rarr; lesson &rarr; topic &rarr; resource &rarr; tag network used by the
  GNN resource mapper. Drag nodes, scroll to zoom, click a node for detail.</div>

  <label class="f">View mode</label>
  <select id="modeSel">
    <option value="sample">Cohort sample network</option>
    <option value="student">Single learner ego-network</option>
    <option value="catalogue">Catalogue only (no learners)</option>
  </select>

  <label class="f">Focus a learner</label>
  <input type="text" id="q" placeholder="Student ID or name, e.g. ST1008" autocomplete="off">
  <button id="go">Load learner graph</button>
  <div class="err" id="err">No learner matched that search.</div>

  <label class="f">Learners in sample</label>
  <input type="range" id="nStud" min="10" max="200" step="10" value="80" style="width:100%">
  <div class="stat"><span>Sampled learners</span><b id="nStudV">80</b></div>

  <label class="f">Entity filter</label>
  <div id="filters"></div>

  <div class="row"><button class="ghost" id="reheat">Re-layout</button><button class="ghost" id="reset">Reset view</button></div>
  <button class="ghost" id="pause">Pause physics</button>

  <label class="f">Visible graph</label>
  <div class="stat"><span>Nodes</span><b id="sN">0</b></div>
  <div class="stat"><span>Edges</span><b id="sE">0</b></div>
  <label class="f">Full knowledge graph</label>
  <div id="fullStats"></div>
  <div id="info"></div>
</div>
<div id="canvasWrap"><canvas id="cv"></canvas><div id="tip"></div>
  <div id="mode"></div>
  <div id="hint">scroll = zoom &middot; drag background = pan &middot; drag node = pin &middot; click = detail</div>
</div></div>
<script>
const DATA = __DATA__, STATS = __STATS__;
const KIND_COLOR = ["#4b93d1","#e0574f","#e0a11b","#2fb086","#7d919a","#a06bd0","#22b8cf"];
const KIND_SIZE  = [5.5,11,8,7,3.4,10,10];
const KIND_LABEL = ["Student","Lesson","Topic","Resource","Tag","Risk level","Learning style"];
const adj = new Map();
DATA.links.forEach(([a,b])=>{ if(!adj.has(a))adj.set(a,[]); if(!adj.has(b))adj.set(b,[]); adj.get(a).push(b); adj.get(b).push(a); });

const show = [true,true,true,true,true,true,true];
const fdiv = document.getElementById("filters");
KIND_LABEL.forEach((name,i)=>{
  const el=document.createElement("label"); el.className="chk";
  el.innerHTML='<input type="checkbox" '+(show[i]?"checked":"")+'><span class="dot" style="background:'+KIND_COLOR[i]+'"></span>'+name;
  el.querySelector("input").onchange=e=>{show[i]=e.target.checked;rebuild();};
  fdiv.appendChild(el);
});
const fs=document.getElementById("fullStats");
[["Nodes",STATS.nodes],["Edges",STATS.edges],["Learners",STATS.students],["Avg degree",STATS.avg_degree],["Components",STATS.components]]
 .forEach(([k,v])=>{const d=document.createElement("div");d.className="stat";d.innerHTML="<span>"+k+"</span><b>"+v+"</b>";fs.appendChild(d);});

let nodes=[], links=[], byId=new Map(), alpha=1, running=true, mode="sample", focusId=null;
const cv=document.getElementById("cv"), ctx=cv.getContext("2d"), tip=document.getElementById("tip");
let W=0,H=0,scale=1,ox=0,oy=0,hover=null,drag=null,panning=false,px=0,py=0;

function resize(){const r=cv.parentElement.getBoundingClientRect();W=r.width;H=r.height;
  const d=window.devicePixelRatio||1;cv.width=W*d;cv.height=H*d;cv.style.width=W+"px";cv.style.height=H+"px";ctx.setTransform(d,0,0,d,0,0);}
window.addEventListener("resize",()=>{resize();});

function studentIds(){const n=+document.getElementById("nStud").value;return DATA.sample.slice(0,n);}

function rebuild(){
  const keep=new Set(), extra=[];
  if(mode==="student" && focusId){
    const s=DATA.students[focusId];
    extra.push({id:focusId,kind:0,label:s[0],meta:{risk:s[1],style:s[2],eng:s[5],perf:s[6],program:s[7]}});
    s[3].forEach(i=>keep.add(i)); s[4].forEach(i=>keep.add(i));
    s[4].forEach(i=>{(adj.get(i)||[]).forEach(j=>keep.add(j));});
    s[3].forEach(i=>{(adj.get(i)||[]).forEach(j=>{if(DATA.nodes[j][1]!==4)keep.add(j);});});
    DATA.nodes.forEach((n,i)=>{ if((n[1]===5&&n[2]===s[1])||(n[1]===6&&n[2]===s[2]))keep.add(i); });
  }else if(mode==="catalogue"){
    DATA.nodes.forEach((n,i)=>{ if(n[1]!==5&&n[1]!==6)keep.add(i); });
  }else{
    DATA.nodes.forEach((n,i)=>keep.add(i));
    studentIds().forEach(sid=>{const s=DATA.students[sid];
      extra.push({id:sid,kind:0,label:s[0],meta:{risk:s[1],style:s[2],eng:s[5],perf:s[6],program:s[7]}});});
  }
  const old=byId; nodes=[]; byId=new Map();
  const push=(id,kind,label,meta)=>{ if(!show[kind]||byId.has(id))return;
    const prev=old.get(id);
    const n={id:id,kind:kind,label:label,meta:meta||{},
      x:prev?prev.x:W/2+(Math.random()-.5)*Math.min(W,700),
      y:prev?prev.y:H/2+(Math.random()-.5)*Math.min(H,600),vx:0,vy:0,deg:0,pin:false};
    byId.set(id,n); nodes.push(n); };
  [...keep].forEach(i=>{const n=DATA.nodes[i];push(n[0],n[1],n[2],n[3]);});
  extra.forEach(e=>push(e.id,e.kind,e.label,e.meta));
  links=[];
  DATA.links.forEach(([a,b,rel])=>{const A=byId.get(DATA.nodes[a][0]),B=byId.get(DATA.nodes[b][0]);
    if(A&&B){links.push({s:A,t:B,rel:rel});A.deg++;B.deg++;}});
  const addStudent=sid=>{const s=DATA.students[sid],A=byId.get(sid);if(!A)return;
    s[3].forEach(i=>{const B=byId.get(DATA.nodes[i][0]);if(B){links.push({s:A,t:B,rel:"NOT_ENGAGED_IN"});A.deg++;B.deg++;}});
    s[4].forEach(i=>{const B=byId.get(DATA.nodes[i][0]);if(B){links.push({s:A,t:B,rel:"RECOMMENDED"});A.deg++;B.deg++;}});
    DATA.nodes.forEach((n,i)=>{ if((n[1]===5&&n[2]===s[1])||(n[1]===6&&n[2]===s[2])){const B=byId.get(n[0]);
      if(B){links.push({s:A,t:B,rel:n[1]===5?"HAS_RISK":"LEARNS_AS"});A.deg++;B.deg++;}}});};
  if(mode==="student"&&focusId)addStudent(focusId); else if(mode==="sample")studentIds().forEach(addStudent);
  document.getElementById("sN").textContent=nodes.length;
  document.getElementById("sE").textContent=links.length;
  document.getElementById("mode").textContent = mode==="student"&&focusId
    ? "Ego-network - "+focusId+" \u00b7 "+DATA.students[focusId][0]+" \u00b7 "+DATA.students[focusId][1]
    : (mode==="catalogue"?"Catalogue sub-graph (lessons, topics, resources, tags)"
       :"Cohort sample - "+studentIds().length+" learners across all risk bands");
  alpha=1;
}

function tick(){
  if(running&&alpha>0.005){
    const n=nodes.length, rep=mode==="student"?2600:1500;
    for(let i=0;i<n;i++){const a=nodes[i];
      for(let j=i+1;j<n;j++){const b=nodes[j];
        let dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy;
        if(d2<1){dx=Math.random()-.5;dy=Math.random()-.5;d2=1;}
        if(d2>90000)continue;
        const f=rep/d2, d=Math.sqrt(d2), fx=f*dx/d, fy=f*dy/d;
        a.vx-=fx;a.vy-=fy;b.vx+=fx;b.vy+=fy;}}
    links.forEach(l=>{const dx=l.t.x-l.s.x,dy=l.t.y-l.s.y,d=Math.sqrt(dx*dx+dy*dy)||1;
      const rest=l.rel==="TAGGED_AS"?34:(l.rel==="RECOMMENDED"?95:70);
      const k=0.016*(d-rest),fx=k*dx/d,fy=k*dy/d;
      l.s.vx+=fx;l.s.vy+=fy;l.t.vx-=fx;l.t.vy-=fy;});
    nodes.forEach(nd=>{ if(nd.pin){nd.vx=0;nd.vy=0;return;}
      nd.vx+=(W/2-nd.x)*0.0016; nd.vy+=(H/2-nd.y)*0.0016;
      nd.vx*=0.86; nd.vy*=0.86;
      nd.x+=Math.max(-14,Math.min(14,nd.vx))*alpha*2.1;
      nd.y+=Math.max(-14,Math.min(14,nd.vy))*alpha*2.1;});
    alpha*=0.985;
  }
  draw(); requestAnimationFrame(tick);
}

function draw(){
  ctx.clearRect(0,0,W,H); ctx.save(); ctx.translate(ox,oy); ctx.scale(scale,scale);
  links.forEach(l=>{
    const hot=hover&&(l.s===hover||l.t===hover);
    ctx.strokeStyle=hot?"rgba(79,209,197,.85)":(l.rel==="RECOMMENDED"?"rgba(47,176,134,.42)":"rgba(140,175,186,.17)");
    ctx.lineWidth=hot?1.7/scale:(l.rel==="RECOMMENDED"?1.2/scale:0.7/scale);
    ctx.beginPath();ctx.moveTo(l.s.x,l.s.y);ctx.lineTo(l.t.x,l.t.y);ctx.stroke();});
  nodes.forEach(nd=>{
    const r=(KIND_SIZE[nd.kind]+Math.min(nd.deg*0.16,5))/1;
    ctx.beginPath();ctx.arc(nd.x,nd.y,r,0,6.2832);
    ctx.fillStyle=KIND_COLOR[nd.kind];ctx.globalAlpha=hover&&hover!==nd?0.65:1;ctx.fill();ctx.globalAlpha=1;
    if(nd===hover||nd.id===focusId){ctx.lineWidth=2/scale;ctx.strokeStyle="#fff";ctx.stroke();}
    const big=nd.kind===1||nd.kind===5||nd.kind===6;
    if(big||nd===hover||(nd.kind===3&&scale>1.1)||(nd.kind===2&&scale>1.35)||(nd.kind===0&&(scale>1.7||mode==="student"))){
      ctx.fillStyle=big?"#eaf3f5":"#b7cdd4";
      ctx.font=(big?700:400)+" "+(big?11.5:9.5)/scale+"px -apple-system,Segoe UI,Roboto,sans-serif";
      ctx.textAlign="center";
      const t=nd.label.length>34?nd.label.slice(0,33)+"\u2026":nd.label;
      ctx.fillText(t,nd.x,nd.y-r-4/scale);}});
  ctx.restore();
}

function pick(mx,my){
  const x=(mx-ox)/scale,y=(my-oy)/scale;let best=null,bd=1e9;
  nodes.forEach(nd=>{const d=(nd.x-x)**2+(nd.y-y)**2, r=(KIND_SIZE[nd.kind]+6)**2;
    if(d<r&&d<bd){bd=d;best=nd;}});return best;
}
cv.addEventListener("mousemove",e=>{
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  if(drag){drag.x=(mx-ox)/scale;drag.y=(my-oy)/scale;drag.pin=true;return;}
  if(panning){ox+=mx-px;oy+=my-py;px=mx;py=my;return;}
  hover=pick(mx,my);
  if(hover){cv.style.cursor="pointer";tip.style.display="block";
    tip.style.left=Math.min(mx+14,W-270)+"px";tip.style.top=(my+14)+"px";tip.innerHTML=tipHtml(hover);}
  else{cv.style.cursor="grab";tip.style.display="none";}
});
function tipHtml(n){
  const k=KIND_LABEL[n.kind],m=n.meta||{};let s="<b>"+n.label+"</b><br><span style='color:#7f9aa3'>"+k+"</span>";
  if(n.kind===3)s+="<br>"+m.lesson+" \u00b7 "+m.topic+"<br>"+m.type+" \u00b7 "+m.min+" min \u00b7 "+m.level;
  if(n.kind===0)s+="<br>"+n.id+" \u00b7 "+m.program+"<br>"+m.risk+" \u00b7 "+m.style+"<br>engagement "+m.eng+" \u00b7 performance "+m.perf;
  s+="<br><span style='color:#7f9aa3'>degree "+n.deg+"</span>";return s;
}
cv.addEventListener("mousedown",e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const n=pick(mx,my);if(n){drag=n;}else{panning=true;px=mx;py=my;cv.style.cursor="grabbing";}});
window.addEventListener("mouseup",()=>{drag=null;panning=false;cv.style.cursor="grab";});
cv.addEventListener("click",e=>{const r=cv.getBoundingClientRect();const n=pick(e.clientX-r.left,e.clientY-r.top);
  if(!n)return;showInfo(n);
  if(n.kind===0){focusId=n.id;mode="student";document.getElementById("modeSel").value="student";document.getElementById("q").value=n.id;rebuild();}});
cv.addEventListener("wheel",e=>{e.preventDefault();const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const f=e.deltaY<0?1.12:1/1.12,ns=Math.max(0.25,Math.min(4.5,scale*f));
  ox=mx-(mx-ox)*(ns/scale);oy=my-(my-oy)*(ns/scale);scale=ns;},{passive:false});

function showInfo(n){
  const el=document.getElementById("info"),m=n.meta||{};el.style.display="block";
  if(n.kind===3){el.innerHTML="<h4>"+n.label+"</h4>"+m.lesson+" \u00b7 "+m.topic+"<br><span class='pill'>"+m.type+
    "</span><span class='pill'>"+m.level+"</span><span class='pill'>"+m.min+" min</span><br><a href='"+m.url+"' target='_blank'>Open in LMS \u2197</a>";}
  else if(n.kind===0){const s=DATA.students[n.id];
    el.innerHTML="<h4>"+s[0]+"</h4>"+n.id+" \u00b7 "+s[7]+"<br><span class='pill'>"+s[1]+"</span><span class='pill'>"+s[2]+
    "</span><br>engagement "+s[5]+" / performance "+s[6]+"<br>gap lessons: "+(s[3].length?s[3].map(i=>DATA.nodes[i][2]).join(", "):"none")+
    "<br>top resources: "+s[4].map(i=>DATA.nodes[i][0]).join(", ");}
  else{const neigh=links.filter(l=>l.s===n||l.t===n).slice(0,8).map(l=>(l.s===n?l.t:l.s).label);
    el.innerHTML="<h4>"+n.label+"</h4>"+KIND_LABEL[n.kind]+" \u00b7 degree "+n.deg+"<br><span style='color:#7f9aa3'>"+neigh.join(", ")+"</span>";}
}

function findStudent(q){
  q=(q||"").trim().toLowerCase();if(!q)return null;
  if(DATA.students[q.toUpperCase()])return q.toUpperCase();
  for(const k in DATA.students){if(DATA.students[k][0].toLowerCase().includes(q))return k;}
  return null;
}
document.getElementById("go").onclick=()=>{
  const id=findStudent(document.getElementById("q").value);
  document.getElementById("err").style.display=id?"none":"block";
  if(id){focusId=id;mode="student";document.getElementById("modeSel").value="student";rebuild();showInfo({kind:0,id:id,label:DATA.students[id][0],meta:{}});}
};
document.getElementById("q").addEventListener("keydown",e=>{if(e.key==="Enter")document.getElementById("go").click();});
document.getElementById("modeSel").onchange=e=>{mode=e.target.value;
  if(mode==="student"&&!focusId){const id=findStudent(document.getElementById("q").value)||DATA.sample[0];focusId=id;}
  rebuild();};
document.getElementById("nStud").oninput=e=>{document.getElementById("nStudV").textContent=e.target.value;if(mode==="sample")rebuild();};
document.getElementById("reheat").onclick=()=>{nodes.forEach(n=>n.pin=false);alpha=1;};
document.getElementById("reset").onclick=()=>{scale=1;ox=0;oy=0;alpha=1;};
document.getElementById("pause").onclick=e=>{running=!running;e.target.textContent=running?"Pause physics":"Resume physics";};

resize();
const qp=new URLSearchParams(location.search).get("student");
if(qp){const id=findStudent(qp);if(id){focusId=id;mode="student";document.getElementById("modeSel").value="student";document.getElementById("q").value=id;}}
rebuild();tick();
</script></body></html>
"""
