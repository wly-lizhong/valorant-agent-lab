const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function harness(){
 const handlers={},timers=[],sent=[],elements={};let now=0,rfb,socket;
 const classes=()=>({add(){},remove(){}});
 const el=id=>elements[id]??=({value:'lotus_easy',classList:classes(),addEventListener(){},replaceChildren(){},requestPointerLock:async()=>{document.pointerLockElement=el('screen');handlers.pointerlockchange();}});
 const document={getElementById:el,createElement:()=>({}),body:{classList:classes()},addEventListener:(n,f)=>handlers[n]=f,exitPointerLock(){document.pointerLockElement=null;handlers.pointerlockchange();},documentElement:{}};
 class RFB {constructor(){rfb=this;this.events={};}addEventListener(n,f){this.events[n]=f;}disconnect(){}}
 class WebSocket {static OPEN=1;constructor(){socket=this;this.events={};this.readyState=1;}addEventListener(n,f){this.events[n]=f;}send(s){sent.push(JSON.parse(s));}close(){}}
 const source=fs.readFileSync(__dirname+'/index.html','utf8').split('<script type="module">')[1].split('</script>')[0].replace(/^import .*;$/m,'');
 vm.runInNewContext(source,{RFB,WebSocket,document,location:{host:'localhost:16080'},window:{addEventListener:(n,f)=>handlers[n]=f},performance:{now:()=>now},setInterval:f=>timers.push(f),fetch:async()=>({ok:true,json:async()=>[{id:'lotus_easy',title:'入门',hint:'提示'}]})});
 return {handlers,timers,sent,el,rfb,socket,document,now:v=>now=v,message:v=>socket.events.message({data:JSON.stringify(v)})};
}
const settle=()=>new Promise(resolve=>setImmediate(resolve));
test('select/start bypasses menus; loading ignores motion; Esc stops game and restores cursor',async()=>{
 const h=harness();await settle();h.rfb.events.connect();h.message({ready:true});
 assert.equal(h.rfb.viewOnly,true);assert.equal(h.el('start').disabled,false);
 await h.el('start').onclick();assert.deepEqual(h.sent.at(-1),{type:'start',course:'lotus_easy'});
 h.handlers.mousemove({movementX:12,movementY:5});h.timers[1]();assert.equal(h.sent.filter(x=>x.type==='move').length,0);
 h.message({launched:true});h.now(120);
 h.handlers.mousemove({movementX:12,movementY:-5});h.timers[1]();assert.deepEqual(h.sent.at(-1),{type:'move',dx:12,dy:-5});
 h.handlers.mousemove({movementX:0,movementY:9000});h.timers[1]();assert.equal(h.sent.filter(x=>x.type==='move').length,1);
 h.document.exitPointerLock();assert.equal(h.el('start').disabled,false);assert.equal(h.rfb.viewOnly,true);assert(h.sent.some(x=>x.type==='stop'));
 h.message({launched:true});assert.deepEqual(h.sent.at(-1),{type:'stop'});
 assert(!h.sent.some(x=>x.key==='Escape'));
});
test('lock failure never launches game; blur cancels startup',async()=>{
 const h=harness();await settle();h.rfb.events.connect();h.message({ready:true});
 const lock=h.el('screen').requestPointerLock;h.el('screen').requestPointerLock=async()=>{throw Error();};
 await h.el('start').onclick();assert(!h.sent.some(x=>x.type==='start'));assert.equal(h.rfb.viewOnly,true);
 h.el('screen').requestPointerLock=lock;await h.el('start').onclick();h.handlers.blur();assert(h.sent.some(x=>x.type==='stop'));assert.equal(h.document.pointerLockElement,null);
});
