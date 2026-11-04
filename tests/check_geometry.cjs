// Run the actual generated diagram code without a browser. This is a geometry
// test, not a screenshot test: DOM layout is stubbed, route coordinates are real.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
  constructor(tag){this.tag=tag;this.attrs={};this.style={};this.children=[];this.hidden=false;this.textContent='';this.dataset={};this.clientWidth=1600;this.clientHeight=900;
    this.classes=new Set();this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),contains:c=>this.classes.has(c),toggle:(c,on)=>on===undefined?(this.classes.has(c)?this.classes.delete(c):this.classes.add(c)):on?this.classes.add(c):this.classes.delete(c)}}
  setAttribute(k,v){this.attrs[k]=String(v)}
  getAttribute(k){return this.attrs[k]||''}
  append(e){this.children.push(e);e.parent=this}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(e=>e!==this)}
  addEventListener(){}
  getBoundingClientRect(){return {x:0,y:0,left:0,top:0,width:1600,height:900}}
  getBBox(){return {x:0,y:0,width:1600,height:900}}
  getContext(){return {measureText:t=>({width:t.length*6})}}
}
const elements=new Map(),timers=[],warnings=[];
const document={querySelector:s=>{if(!elements.has(s))elements.set(s,new Element(s));return elements.get(s)},
  createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),
  querySelectorAll:()=>[],addEventListener:()=>{}};
const context=vm.createContext({document,console:{log:console.log,warn:(...a)=>warnings.push(a)},window:{addEventListener:()=>{}},
  setTimeout:f=>timers.push(f),requestAnimationFrame:f=>timers.push(f),cancelAnimationFrame:()=>{},Set,Map,TextEncoder,Blob});
const html=fs.readFileSync(process.argv[2],'utf8'),script=html.match(/<script>([\s\S]*)<\/script>/)[1];
vm.runInContext(script,context,{timeout:60000});
for(const f of timers.splice(0))f();
const inspect=()=>vm.runInContext(`(()=>{
  const records=edgeRecords.filter(r=>edgeVisible(r.edge)),errors=[];
  const segments=[];
  for(const r of records){
    if(!r.points){errors.push('Missing route '+r.edge.source+' -> '+r.edge.target);continue}
    if(!pathDirectionsValid(r.points,r.edge._sourceSide,r.edge._targetSide))errors.push('Bad terminal '+r.edge.source);
    for(let i=1;i<r.points.length;i++){
      const a=r.points[i-1],b=r.points[i];
      if(a.x!==b.x&&a.y!==b.y)errors.push('Diagonal');
      if(routeObstacles(r.edge).some(box=>segmentHitsBox(a,b,box)))errors.push('Block collision '+r.edge.source+' -> '+r.edge.target);
      for(const s of segments)if(s.record!==r&&segmentsOverlap(a,b,s.a,s.b,1))errors.push('Merged '+r.edge.source+' -> '+r.edge.target+' with '+s.record.edge.source+' -> '+s.record.edge.target);
      segments.push({a,b,record:r});
    }
  }
  const boxes=visibleRoutingBoxes(),points=records.flatMap(r=>r.points||[]);
  const minY=Math.min(...boxes.map(b=>b.y)),maxY=Math.max(...boxes.map(b=>b.y+b.h)),minX=Math.min(...boxes.map(b=>b.x)),maxX=Math.max(...boxes.map(b=>b.x+b.w));
  const detour=Math.max(0,...points.map(p=>Math.max(minX-p.x,p.x-maxX,minY-p.y,p.y-maxY)));
  if(detour>400)errors.push('Excessive exterior detour '+detour);
  return {errors,edges:records.length,detour,boxes:boxes.map(b=>({x:b.x,y:b.y,w:b.w,h:b.h,id:b.id||b.group?.name})),
    groups:stageRecords.filter(g=>!g.hidden&&!g.collapsed).map(g=>({...g.bounds,name:g.name})),
    routes:records.map(r=>({source:r.edge.source,target:r.edge.target,points:r.points,label:r.label?.textContent,labelX:r.label?.getAttribute('x'),labelY:r.label?.getAttribute('y')}))};
})()`,context);
const initial=inspect();
console.log(JSON.stringify({scenario:'initial',edges:initial.edges,detour:initial.detour,errors:initial.errors}));
if(process.argv[3])fs.writeFileSync(process.argv[3],JSON.stringify(initial,null,2));
if(process.argv[4]){
  const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
  const serialize=e=>e.style.display==='none'?'':`<${e.tag} ${Object.entries(e.attrs).filter(([k])=>k!=='transform'||e!==elements.get('#viewport')).map(([k,v])=>`${k}="${esc(v)}"`).join(' ')}>${esc(e.textContent)}${e.children.map(serialize).join('')}</${e.tag}>`;
  const points=initial.routes.flatMap(r=>r.points||[]).concat(initial.boxes.flatMap(b=>[{x:b.x,y:b.y},{x:b.x+b.w,y:b.y+b.h}]));
  const x=Math.min(...points.map(p=>p.x))-40,y=Math.min(...points.map(p=>p.y))-40,w=Math.max(...points.map(p=>p.x))-x+40,h=Math.max(...points.map(p=>p.y))-y+40;
  const vp=elements.get('#viewport');vp.tag='g';
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="${x} ${y} ${w} ${h}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#1d2733"/></marker><marker id="arrow-structural" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#76529b"/></marker></defs><rect x="${x}" y="${y}" width="${w}" height="${h}" fill="white"/>${serialize(vp)}</svg>`;
  fs.writeFileSync(process.argv[4],svg);
}
assert.deepEqual(initial.errors,[]);
const persistence=vm.runInContext(`(()=>{const id=graph.nodes[0].id,expected={...pos[id]},expectedView={scale,tx,ty},saved=exportLayoutState();pos[id].x+=321;scale=.42;tx=777;ty=-123;const restored=importLayoutState(saved);return{restored,position:{...pos[id]},expected,view:{scale,tx,ty},expectedView}})()`,context);
assert.equal(persistence.restored,true);
assert.deepEqual(persistence.position,persistence.expected,'Saved node positions must restore');
assert.deepEqual(persistence.view,persistence.expectedView,'Saved pan and zoom must restore');
const hover=vm.runInContext(`(()=>{const r=edgeRecords.find(record=>record.points);r.hit.onpointerenter();const active=r.path.classList.contains('hovered')&&r.label.classList.contains('hovered')&&r.path.getAttribute('marker-end')==='url(#arrow-highlight)'&&r.hit.getAttribute('d')===r.path.getAttribute('d');r.hit.onpointerleave();return{active,cleared:!r.path.classList.contains('hovered')&&r.path.getAttribute('marker-end')===r.marker}})()`,context);
assert.deepEqual(hover,{active:true,cleared:true},'Arrow hover must highlight and restore the complete route');
vm.runInContext(`compactLayout(false)`,context);
assert.deepEqual(inspect().errors,[]);
assert.equal(JSON.stringify(inspect().boxes),JSON.stringify(initial.boxes),'Compact must be idempotent');
vm.runInContext(`resetLayout()`,context);
assert.deepEqual(inspect().errors,[]);
assert.equal(JSON.stringify(inspect().boxes),JSON.stringify(initial.boxes),'Reset must reproduce initial compact layout');
vm.runInContext(`const firstCollapsed=stageRecords.find(g=>g.collapsed);if(firstCollapsed){firstCollapsed.collapsedPosition.y+=220;refreshLayout()}`,context);
const moved=inspect();console.log(JSON.stringify({scenario:'moved',edges:moved.edges,detour:moved.detour,errors:moved.errors}));
assert.deepEqual(moved.errors,[]);
vm.runInContext(`resetLayout();const expanded=stageRecords.find(g=>g.collapsed);if(expanded){expanded.collapsed=false;compactLayout(false)}`,context);
const expanded=inspect();console.log(JSON.stringify({scenario:'expanded',edges:expanded.edges,detour:expanded.detour,errors:expanded.errors}));
assert.deepEqual(expanded.errors,[]);
vm.runInContext(`resetLayout();stageRecords.forEach(g=>g.collapsed=false);compactLayout(false)`,context);
const allExpanded=inspect();console.log(JSON.stringify({scenario:'all expanded',edges:allExpanded.edges,detour:allExpanded.detour,errors:allExpanded.errors}));
assert.deepEqual(allExpanded.errors,[]);
vm.runInContext(`resetLayout();toggleLossVisibility();compactLayout(false)`,context);
const noLoss=inspect();console.log(JSON.stringify({scenario:'loss hidden',edges:noLoss.edges,detour:noLoss.detour,errors:noLoss.errors}));
assert.deepEqual(noLoss.errors,[]);
console.log('Geometry checks passed');
