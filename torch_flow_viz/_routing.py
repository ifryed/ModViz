"""Orthogonal routing embedded in the self-contained HTML renderer."""

ROUTING_JS = r'''
const ROUTE_CLEARANCE=6, LANE_SPACE=14;
let routedSegments=[];
function groupForNode(id){return stageRecords.find(r=>r.members.some(n=>n.id===id))||null}
function boxForNode(id){const g=groupForNode(id);return g?.collapsed?{...g.bounds,group:g}:{...pos[id],w:W,h:H,group:null}}
function sideVector(side){return {left:{x:-1,y:0},right:{x:1,y:0},top:{x:0,y:-1},bottom:{x:0,y:1}}[side]}
function edgePort(p,side,offset=0){const w=p.w||W,h=p.h||H;return side==='left'?{x:p.x,y:p.y+h/2+offset}:side==='right'?{x:p.x+w,y:p.y+h/2+offset}:side==='top'?{x:p.x+w/2+offset,y:p.y}:{x:p.x+w/2+offset,y:p.y+h}}
function simplifyRoute(points){
  const result=[];
  for(const p of points){
    if(result.length&&p.x===result.at(-1).x&&p.y===result.at(-1).y)continue;
    while(result.length>1){
      const a=result.at(-2),b=result.at(-1);
      // Only remove a middle point when it lies BETWEEN its neighbours.
      if(!((a.x===b.x&&b.x===p.x&&(b.y-a.y)*(p.y-b.y)>=0)||
           (a.y===b.y&&b.y===p.y&&(b.x-a.x)*(p.x-b.x)>=0)))break;
      result.pop();
    }
    result.push({...p});
  }
  return result;
}
function segmentsOverlap(a,b,c,d,clearance=ROUTE_CLEARANCE){
  if(a.y===b.y&&c.y===d.y&&Math.abs(a.y-c.y)<clearance)
    return Math.min(Math.max(a.x,b.x),Math.max(c.x,d.x))-Math.max(Math.min(a.x,b.x),Math.min(c.x,d.x))>.01;
  if(a.x===b.x&&c.x===d.x&&Math.abs(a.x-c.x)<clearance)
    return Math.min(Math.max(a.y,b.y),Math.max(c.y,d.y))-Math.max(Math.min(a.y,b.y),Math.min(c.y,d.y))>.01;
  return false;
}
function segmentConflicts(a,b){return routedSegments.some(([c,d])=>segmentsOverlap(a,b,c,d))}
function segmentHitsBox(a,b,r){
  return a.y===b.y?a.y>r.y+.01&&a.y<r.y+r.h-.01&&Math.min(a.x,b.x)<r.x+r.w-.01&&Math.max(a.x,b.x)>r.x+.01:
    a.x===b.x?a.x>r.x+.01&&a.x<r.x+r.w-.01&&Math.min(a.y,b.y)<r.y+r.h-.01&&Math.max(a.y,b.y)>r.y+.01:true;
}
function visibleRoutingBoxes(exclude=[]){
  const omitted=new Set(exclude),boxes=[];
  for(const n of graph.nodes)if(!omitted.has(n.id)&&(showLoss||n.category!=='loss')&&!groupForNode(n.id)?.collapsed)
    boxes.push({...pos[n.id],w:W,h:H,id:n.id});
  for(const g of stageRecords)if(!g.hidden&&g.collapsed&&!g.members.some(n=>omitted.has(n.id)))boxes.push({...g.bounds,group:g});
  return boxes;
}
function routeObstacles(e){
  const sg=groupForNode(e.source),tg=groupForNode(e.target);
  return [...visibleRoutingBoxes(),...stageRecords.filter(g=>!g.hidden&&!g.collapsed&&g!==sg&&g!==tg).map(g=>({...g.bounds,group:g}))];
}
function edgeVisible(e){const a=groupForNode(e.source),b=groupForNode(e.target);return(showLoss||(byId[e.source].category!=='loss'&&byId[e.target].category!=='loss'))&&!(a?.collapsed&&a===b)}
function endpointKey(id){const g=groupForNode(id);return g?.collapsed?`group:${g.name}`:`node:${id}`}
function updateEdgeSides(){
  const ports=new Map();
  for(const e of graph.edges){
    if(!edgeVisible(e))continue;
    const a=boxForNode(e.source),b=boxForNode(e.target),dx=b.x+b.w/2-a.x-a.w/2,dy=b.y+b.h/2-a.y-a.h/2;
    e._sourceSide=Math.abs(dx)>=Math.abs(dy)?(dx>=0?'right':'left'):(dy>=0?'bottom':'top');
    e._targetSide={left:'right',right:'left',top:'bottom',bottom:'top'}[e._sourceSide];
    // Long bypass connections leave via the top/bottom instead of aiming through a row.
    const direct=[edgePort(a,e._sourceSide),edgePort(b,e._targetSide)];
    const blocked=visibleRoutingBoxes([e.source,e.target]).some(r=>segmentHitsBox(direct[0],direct[1],r));
    if(Math.abs(dx)>a.w+b.w&&Math.abs(dy)<Math.max(a.h,b.h)&&blocked){
      e._sourceSide=e._targetSide='top';
    }
    for(const source of [true,false]){
      const id=source?e.source:e.target,side=e[source?'_sourceSide':'_targetSide'],key=`${endpointKey(id)}:${side}`;
      if(!ports.has(key))ports.set(key,[]);
      ports.get(key).push({e,source,box:source?a:b,other:source?b:a});
    }
  }
  for(const entries of ports.values()){
    const horizontal=['top','bottom'].includes(entries[0].e[entries[0].source?'_sourceSide':'_targetSide']);
    entries.sort((a,b)=>(horizontal?a.other.x-b.other.x:a.other.y-b.other.y)||graph.edges.indexOf(a.e)-graph.edges.indexOf(b.e));
    const size=horizontal?entries[0].box.w:entries[0].box.h,step=Math.min(LANE_SPACE,(size-20)/Math.max(1,entries.length-1));
    entries.forEach((p,i)=>p.e[p.source?'_sourcePortOffset':'_targetPortOffset']=(i-(entries.length-1)/2)*step);
  }
}
function endpointPortOffset(e,source){return e[source?'_sourcePortOffset':'_targetPortOffset']||0}
function pathDirectionsValid(points,sourceSide,targetSide){
  if(points.length<2)return false;
  const a=points[0],b=points[1],c=points.at(-2),d=points.at(-1),s=sideVector(sourceSide),t=sideVector(targetSide);
  return (b.x-a.x)*s.x+(b.y-a.y)*s.y>0&&(c.x-d.x)*t.x+(c.y-d.y)*t.y>0&&
    (s.x?b.y===a.y:b.x===a.x)&&(t.x?c.y===d.y:c.x===d.x);
}
class RouteHeap{
  constructor(){this.data=[]}
  push(v){const a=this.data;a.push(v);let i=a.length-1;while(i){const p=(i-1)>>1;if(a[p].f<=v.f)break;a[i]=a[p];i=p}a[i]=v}
  pop(){const a=this.data,first=a[0],last=a.pop();if(a.length){let i=0;while(i*2+1<a.length){let c=i*2+1;if(c+1<a.length&&a[c+1].f<a[c].f)c++;if(a[c].f>=last.f)break;a[i]=a[c];i=c}a[i]=last}return first}
}
function routeEdge(e){
  const a=boxForNode(e.source),b=boxForNode(e.target),ss=e._sourceSide,ts=e._targetSide,
    start=edgePort(a,ss,endpointPortOffset(e,true)),goal=edgePort(b,ts,endpointPortOffset(e,false)),
    sv=sideVector(ss),tv=sideVector(ts),lead=18,
    s={x:start.x+sv.x*lead,y:start.y+sv.y*lead},t={x:goal.x+tv.x*lead,y:goal.y+tv.y*lead},
    obstacles=routeObstacles(e),clear=(u,v)=>!segmentConflicts(u,v)&&!obstacles.some(r=>segmentHitsBox(u,v,r));
  const valid=p=>pathDirectionsValid(p,ss,ts)&&p.slice(1).every((v,i)=>clear(p[i],v));
  const simple=[[start,goal],[start,s,{x:t.x,y:s.y},t,goal],[start,s,{x:s.x,y:t.y},t,goal]].map(simplifyRoute);
  for(const p of simple)if(valid(p))return p;
  // Exact endpoint coordinates avoid snapping different arrows onto a shared grid cell.
  // Existing routes contribute neighbouring lanes, allowing crossings but forbidding overlap.
  const xs=new Set([s.x,t.x]),ys=new Set([s.y,t.y]);
  for(const r of obstacles){xs.add(r.x-18);xs.add(r.x+r.w+18);ys.add(r.y-18);ys.add(r.y+r.h+18)}
  for(const [u,v] of routedSegments){
    if(u.x===v.x){xs.add(u.x-LANE_SPACE);xs.add(u.x+LANE_SPACE)}
    else{ys.add(u.y-LANE_SPACE);ys.add(u.y+LANE_SPACE)}
  }
  const x=[...xs].sort((a,b)=>a-b),y=[...ys].sort((a,b)=>a-b),nx=x.length,ny=y.length,
    si=x.indexOf(s.x),sj=y.indexOf(s.y),ti=x.indexOf(t.x),tj=y.indexOf(t.y),
    state=(i,j,dir)=>((j*nx+i)*2+dir),heap=new RouteHeap(),cost=new Map(),parent=new Map(),
    point=(i,j)=>({x:x[i],y:y[j]}),cache=new Map();
  const startDir=sv.x?0:1,startKey=state(si,sj,startDir);
  cost.set(startKey,0);heap.push({i:si,j:sj,dir:startDir,g:0,f:0,key:startKey});
  let end=null;
  while(heap.data.length){
    const c=heap.pop();if(c.g!==cost.get(c.key))continue;
    if(c.i===ti&&c.j===tj){end=c;break}
    for(const [di,dj,dir] of [[1,0,0],[-1,0,0],[0,1,1],[0,-1,1]]){
      const i=c.i+di,j=c.j+dj;if(i<0||i>=nx||j<0||j>=ny)continue;
      const u=point(c.i,c.j),v=point(i,j),aKey=c.j*nx+c.i,bKey=j*nx+i,ck=`${Math.min(aKey,bKey)}:${Math.max(aKey,bKey)}`;
      if(!cache.has(ck))cache.set(ck,clear(u,v));
      if(!cache.get(ck))continue;
      const g=c.g+Math.abs(u.x-v.x)+Math.abs(u.y-v.y)+(dir!==c.dir?28:0),key=state(i,j,dir);
      if(g>=(cost.get(key)??Infinity))continue;
      cost.set(key,g);parent.set(key,c.key);
      heap.push({i,j,dir,g,key,f:g+Math.abs(v.x-t.x)+Math.abs(v.y-t.y)});
    }
  }
  if(!end)return null; // Never draw an unchecked route through blocks or other arrows.
  const path=[];let cursor=end.key;
  while(cursor!==undefined){const cell=Math.floor(cursor/2);path.push(point(cell%nx,Math.floor(cell/nx)));cursor=parent.get(cursor)}
  const result=simplifyRoute([start,...path.reverse(),goal]);
  return valid(result)?result:null;
}
function drawEdge(record,points){
  record.points=points;
  const d=points.map((p,i)=>`${i?'L':'M'}${p.x} ${p.y}`).join(' ');
  record.path.setAttribute('d',d);record.hit.setAttribute('d',d);
  if(record.label){
    record.label.setAttribute('transform','');
    const candidates=points.slice(1).map((b,i)=>({a:points[i],b,length:Math.abs(b.x-points[i].x)})).filter(s=>s.a.y===s.b.y).sort((a,b)=>b.length-a.length);
    const s=candidates[0];
    record.label.setAttribute('x',s?(s.a.x+s.b.x)/2:points[0].x+12);
    record.label.setAttribute('y',s?s.a.y-6:(points[0].y+points.at(-1).y)/2);
  }
}
function updateEdges(){
  updateEdgeSides();routedSegments=[];
  // Reserve short local connections before bypass routes.
  const ordered=[...edgeRecords].sort((r,q)=>{
    const distance=e=>{const a=boxForNode(e.source),b=boxForNode(e.target);return Math.abs(a.x-b.x)+Math.abs(a.y-b.y)};
    return distance(r.edge)-distance(q.edge);
  });
  for(const record of ordered){
    const visible=edgeVisible(record.edge);
    record.path.style.display=visible?'':'none';record.hit.style.display=visible?'':'none';if(record.label)record.label.style.display=visible?'':'none';
    record.points=null;if(!visible)continue;
    const points=routeEdge(record.edge);
    record.routeFailed=!points;
    if(!points){record.path.setAttribute('d','');record.hit.setAttribute('d','');if(record.label)record.label.style.display='none';console.warn('No clear route',record.edge.source,record.edge.target);continue}
    drawEdge(record,points);
    for(let i=1;i<points.length;i++)routedSegments.push([points[i-1],points[i]]);
  }
}
function updateEdgesFast(){
  // Deform cached routes during a drag; full routing runs once on release.
  for(const r of edgeRecords){
    if(!r.points?.length||r.path.style.display==='none')continue;
    const e=r.edge,points=r.points.map(p=>({...p})),start=edgePort(boxForNode(e.source),e._sourceSide,endpointPortOffset(e,true)),
      goal=edgePort(boxForNode(e.target),e._targetSide,endpointPortOffset(e,false)),oldStart=points[0],oldGoal=points.at(-1);
    const dx=start.x-oldStart.x,dy=start.y-oldStart.y;
    if(Math.abs(dx-(goal.x-oldGoal.x))<.01&&Math.abs(dy-(goal.y-oldGoal.y))<.01)points.forEach(p=>{p.x+=dx;p.y+=dy});
    else{
      const s=sideVector(e._sourceSide),t=sideVector(e._targetSide),sl={x:start.x+s.x*18,y:start.y+s.y*18},tl={x:goal.x+t.x*18,y:goal.y+t.y*18};
      const interior=points.length>2?points.slice(1,-1):[{x:(start.x+goal.x)/2,y:(start.y+goal.y)/2}];
      points.splice(0,points.length,start,sl,{x:interior[0].x,y:sl.y},...interior,{x:interior.at(-1).x,y:tl.y},tl,goal);
    }
    const d=points.map((p,i)=>`${i?'L':'M'}${p.x} ${p.y}`).join(' ');r.path.setAttribute('d',d);r.hit.setAttribute('d',d);
    if(r.label)r.label.setAttribute('transform',`translate(${(dx+goal.x-oldGoal.x)/2} ${(dy+goal.y-oldGoal.y)/2})`);
  }
}
'''
