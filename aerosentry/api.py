"""FastAPI surface: chat, human approval, traces, health, plus a tiny operator console at /."""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agents import Orchestrator
from .config import settings
from .observability import langsmith_enabled, read_traces

app = FastAPI(title="AeroSentry Agents", version="0.1.0")
orch = Orchestrator()


class ChatIn(BaseModel):
    message: str
    thread_id: str | None = None


class ApproveIn(BaseModel):
    thread_id: str
    approved: bool


@app.get("/health")
def health():
    return {"ok": True, "provider": settings.provider, "model": settings.model, "langsmith": langsmith_enabled()}


@app.post("/chat")
def chat(body: ChatIn):
    return orch.run(body.message, body.thread_id)


@app.post("/approve")
def approve(body: ApproveIn):
    return orch.approve(body.thread_id, body.approved)


@app.get("/traces")
def traces(run_id: str | None = None, limit: int = 200):
    return read_traces(run_id, limit)


CONSOLE = """<!doctype html><html><head><meta charset=utf-8><title>AeroSentry console</title>
<style>body{font:14px/1.5 system-ui;margin:0;background:#0f1419;color:#e6edf3}main{max-width:900px;margin:0 auto;padding:24px}
h1{font-size:20px;margin:0 0 12px}textarea,button{font:inherit}textarea{width:100%;height:70px;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:8px}
button{background:#1f6feb;color:#fff;border:0;border-radius:6px;padding:8px 14px;margin:8px 6px 8px 0;cursor:pointer}button.warn{background:#b62324}button.ok{background:#238636}
pre{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;white-space:pre-wrap;word-break:break-word}.tag{display:inline-block;padding:2px 8px;border-radius:10px;background:#30363d;margin-right:6px}</style></head>
<body><main><h1>AeroSentry - agentic operations console</h1>
<textarea id=q placeholder="e.g. Schedule a thermal inspection for D-01 at solar-farm-01 / Inspect data/imagery/solar_thermal_anomaly.png / What is the wind limit for launches?"></textarea>
<div><button onclick=send()>Send</button><button class=ok id=a onclick=decide(true) disabled>Approve</button><button class=warn id=r onclick=decide(false) disabled>Reject</button></div>
<div id=meta></div><pre id=out>ready</pre></main>
<script>let tid=null;const $=s=>document.querySelector(s);
async function post(p,b){const r=await fetch(p,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});return r.json()}
function show(d){tid=d.thread_id;$('#meta').innerHTML=`<span class=tag>route: ${d.route}</span><span class=tag>risk: ${d.risk}</span><span class=tag>pending: ${d.pending_approval}</span><span class=tag>fallback: ${d.fallback_used}</span>`;
$('#out').textContent=JSON.stringify(d,null,2);$('#a').disabled=$('#r').disabled=!d.pending_approval}
async function send(){$('#out').textContent='running...';show(await post('/chat',{message:$('#q').value}))}
async function decide(ok){show(await post('/approve',{thread_id:tid,approved:ok}))}</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def console():
    return CONSOLE
