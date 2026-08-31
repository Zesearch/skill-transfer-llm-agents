#!/usr/bin/env python3
"""Rebuild the trajectory viewer from the released run pickles.

Reads every dialogue pkl in data/results/ (memory snapshots and cache files are skipped), writes one small HTML fragment per task to data/results/inspector_tasks/, and writes the data/inspector.html shell that lists all tasks and shows a fragment in an iframe when clicked. The shell works both over file:// (iframes, no fetch) and via python -m http.server.

Usage, from the repo root:
    python scripts/build_inspector.py            # all dialogue pkls
    python scripts/build_inspector.py --stage final_v1   # subset by stage token
"""
import argparse
import html
import json
import os
import pickle
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "results")
FRAG_DIR = os.path.join(RESULTS, "inspector_tasks")
SHELL = os.path.join(ROOT, "data", "inspector.html")

STAGES = ("final_v1", "subskill_v1", "prompt_abl2_L1", "prompt_abl2_L2")

FRAG_CSS = """
body{font-family:system-ui,sans-serif;margin:12px;font-size:13px;color:#222}
h2{font-size:15px;margin:4px 0 8px}
table.meta{border-collapse:collapse;margin-bottom:10px}
table.meta td{border:1px solid #ddd;padding:2px 8px}
table.meta td:first-child{background:#f6f6f6;font-weight:600}
.msg{border:1px solid #e0e0e0;border-radius:6px;margin:6px 0;overflow:hidden}
.msg .hdr{padding:2px 8px;font-weight:600;font-size:12px}
.msg pre{margin:0;padding:6px 8px;white-space:pre-wrap;word-break:break-word;font-size:12px;background:#fff}
.role-system .hdr{background:#ececff}.role-user .hdr{background:#e7f4e7}
.role-assistant .hdr{background:#fdf1dc}.role-tool .hdr{background:#f3e6f5}
.skill{border:1px dashed #bbb;border-radius:6px;margin:6px 0;padding:4px 8px}
.skill pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#fafafa;margin:4px 0;padding:4px}
details{margin:6px 0}summary{cursor:pointer;font-weight:600}
.ok{color:#1a7f37}.fail{color:#b42318}
"""


def parse_stem(stem):
    """<benchmark>_<agent>_<model>_<stage> -> fields; agent may contain one underscore (linear_asi)."""
    for stage in STAGES:
        if stem.endswith("_" + stage):
            rest = stem[: -len(stage) - 1]
            break
    else:
        return None
    bench, rest = rest.split("_", 1)
    m = re.match(r"(linear|subtask)(_(?:asi|awm))?_(.+)$", rest)
    if not m:
        return None
    agent = m.group(1) + (m.group(2) or "")
    return {"bench": bench, "agent": agent, "model": m.group(3), "stage": stage}


def esc(x):
    return html.escape(str(x), quote=False)


def render_tool_calls(tcs):
    out = []
    for tc in tcs or []:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
        args = tc.get("arguments") if isinstance(tc, dict) else getattr(tc, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                pass
        try:
            pretty = json.dumps(args, indent=1, ensure_ascii=False)
        except (TypeError, ValueError):
            pretty = str(args)
        out.append(f"[tool call] {name}\n{pretty}")
    return "\n".join(out)


def render_message(m):
    role = m.get("role", "?")
    parts = []
    content = m.get("content")
    if content:
        parts.append(esc(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)))
    if m.get("tool_calls"):
        parts.append(esc(render_tool_calls(m["tool_calls"])))
    body = "\n".join(p for p in parts if p) or "(empty)"
    return f'<div class="msg role-{esc(role)}"><div class="hdr">{esc(role)}</div><pre>{body}</pre></div>'


def render_fragment(rec, meta):
    score = rec.get("tgc_score")
    ok = bool(rec.get("success"))
    rows = [
        ("task_id", rec.get("task_id")), ("benchmark", meta["bench"]),
        ("agent", meta["agent"]), ("model", meta["model"]), ("stage", meta["stage"]),
        ("success", f'<span class="{"ok" if ok else "fail"}">{ok}</span>'),
        ("score", "" if score is None else f"{score:.3f}" if isinstance(score, float) else score),
        ("completed", rec.get("completed")), ("env_steps", rec.get("env_steps")),
        ("subtasks", rec.get("subtask_count")),
        ("wall clock (s)", f"{rec.get('wall_clock_seconds', 0):.0f}"),
        ("prompt tokens", rec.get("total_prompt_tokens")),
        ("completion tokens", rec.get("total_completion_tokens")),
        ("difficulty", rec.get("difficulty")),
        ("library size at start", rec.get("library_size_at_start")),
    ]
    meta_html = "".join(f"<tr><td>{esc(k)}</td><td>{v if k == 'success' else esc(v)}</td></tr>"
                        for k, v in rows if v is not None and v != "")
    sections = [f'<h2>{esc(rec.get("task_id"))} &middot; {esc(meta["bench"])} &middot; '
                f'{esc(meta["agent"])} &middot; {esc(meta["model"])} &middot; {esc(meta["stage"])}</h2>',
                f'<table class="meta">{meta_html}</table>']

    plans = rec.get("subtask_plans") or []
    if plans:
        items = "".join(f"<li><pre>{esc(p)}</pre></li>" for p in plans)
        sections.append(f"<details><summary>Subtask plans ({len(plans)})</summary><ol>{items}</ol></details>")
    sums = rec.get("subtask_summaries") or []
    if sums:
        items = "".join(f"<li><pre>{esc(s)}</pre></li>" for s in sums)
        sections.append(f"<details><summary>Subtask summaries ({len(sums)})</summary><ol>{items}</ol></details>")

    retrieved = rec.get("retrieved_skills") or []
    if retrieved:
        items = "".join(f"<li>{esc(s.get('id'))} &middot; {esc(s.get('name', ''))} "
                        f"(from {esc(s.get('source_task_id', '?'))})</li>" for s in retrieved)
        sections.append(f"<details><summary>Retrieved skills ({len(retrieved)})</summary><ul>{items}</ul></details>")
    induced = rec.get("skills_induced") or []
    if induced:
        blocks = "".join(
            f'<div class="skill"><b>{esc(s.get("id"))}</b> {esc(s.get("description", ""))}'
            f'<pre>{esc(s.get("source", ""))}</pre></div>' for s in induced)
        sections.append(f"<details open><summary>Skills induced ({len(induced)})</summary>{blocks}</details>")

    history = rec.get("final_history") or []
    msgs = "".join(render_message(m) for m in history)
    sections.append(f"<h2>Dialogue ({len(history)} messages)</h2>{msgs}")

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{FRAG_CSS}</style></head><body>" + "".join(sections) + "</body></html>")


SHELL_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Trajectory inspector</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;font-size:13px;display:flex;height:100vh}
#left{width:44%;min-width:380px;display:flex;flex-direction:column;border-right:1px solid #ccc}
#controls{padding:8px;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
select,input{font-size:12px;padding:2px}
#count{color:#666;font-size:12px;margin-left:auto}
#list{flex:1;overflow-y:auto}
table{border-collapse:collapse;width:100%}
th,td{padding:2px 6px;border-bottom:1px solid #eee;text-align:left;font-size:12px;white-space:nowrap}
th{position:sticky;top:0;background:#f6f6f6;cursor:default}
tr.row{cursor:pointer}tr.row:hover{background:#f0f6ff}tr.sel{background:#dcebff}
.ok{color:#1a7f37;font-weight:600}.fail{color:#b42318}
#right{flex:1;display:flex;flex-direction:column}
#right .hint{margin:auto;color:#888}
iframe{border:0;width:100%;height:100%}
</style></head><body>
<div id="left">
 <div id="controls">
  <select id="fBench"></select><select id="fAgent"></select>
  <select id="fModel"></select><select id="fStage"></select>
  <select id="fOutcome"><option value="">outcome: all</option>
   <option value="1">success</option><option value="0">failure</option></select>
  <input id="fSearch" placeholder="task id search" size="14">
  <span id="count"></span>
 </div>
 <div id="list"><table><thead><tr>
  <th>benchmark</th><th>task id</th><th>agent</th><th>model</th><th>stage</th><th>outcome</th>
 </tr></thead><tbody id="rows"></tbody></table></div>
</div>
<div id="right"><div class="hint">Click a task to load its trajectory.</div></div>
<script id="INDEX" type="application/json">__INDEX__</script>
<script>
const IDX = JSON.parse(document.getElementById('INDEX').textContent);
const TASKS = [];
for (const g of IDX.groups){
  const ids = IDX.bench_ids[g.bench];
  for (let k = 0; k < g.t.length; k++)
    TASKS.push({b:g.bench,a:g.agent,m:g.model,s:g.stage,stem:g.stem,
                idx:k,id:ids[g.t[k]],ok:g.ok.charCodeAt(k)===49});
}
function fill(sel, label, vals){
  const s=document.getElementById(sel);
  s.innerHTML='<option value="">'+label+': all</option>'+
    [...new Set(vals)].sort().map(v=>'<option>'+v+'</option>').join('');
}
fill('fBench','benchmark',TASKS.map(t=>t.b));
fill('fAgent','agent',TASKS.map(t=>t.a));
fill('fModel','model',TASKS.map(t=>t.m));
fill('fStage','stage',TASKS.map(t=>t.s));
const MAX_ROWS=2000;
let selRow=null;
function render(){
  const b=fBench.value,a=fAgent.value,m=fModel.value,s=fStage.value,o=fOutcome.value,
        q=fSearch.value.trim().toLowerCase();
  const hit=TASKS.filter(t=>(!b||t.b===b)&&(!a||t.a===a)&&(!m||t.m===m)&&(!s||t.s===s)
    &&(!o||String(t.ok?1:0)===o)&&(!q||t.id.toLowerCase().includes(q)));
  document.getElementById('count').textContent=
    hit.length+' / '+TASKS.length+(hit.length>MAX_ROWS?' (first '+MAX_ROWS+' shown)':'');
  const tb=document.getElementById('rows');
  tb.innerHTML='';
  for(const t of hit.slice(0,MAX_ROWS)){
    const tr=document.createElement('tr');
    tr.className='row';
    tr.innerHTML='<td>'+t.b+'</td><td>'+t.id+'</td><td>'+t.a+'</td><td>'+t.m+'</td><td>'+
      t.s+'</td><td class="'+(t.ok?'ok':'fail')+'">'+(t.ok?'success':'failure')+'</td>';
    tr.onclick=()=>{
      if(selRow)selRow.classList.remove('sel');
      selRow=tr;tr.classList.add('sel');
      const right=document.getElementById('right');
      right.innerHTML='';
      const fr=document.createElement('iframe');
      fr.src='results/inspector_tasks/'+t.stem+'__'+String(t.idx).padStart(3,'0')+'.html';
      right.appendChild(fr);
    };
    tb.appendChild(tr);
  }
}
for(const id of ['fBench','fAgent','fModel','fStage','fOutcome'])
  document.getElementById(id).onchange=render;
fSearch.oninput=render;
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="only pkls of this stage token")
    ap.add_argument("--limit", type=int, default=0, help="stop after N pkls (debug)")
    args = ap.parse_args()

    os.makedirs(FRAG_DIR, exist_ok=True)
    stems = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith(".pkl") or f.endswith("_memory.pkl") or f.startswith("_"):
            continue
        stem = f[:-4]
        meta = parse_stem(stem)
        if meta is None:
            print(f"  ! unrecognized name, skipped: {f}", file=sys.stderr)
            continue
        if args.stage and meta["stage"] != args.stage:
            continue
        stems.append((stem, meta))
    if args.limit:
        stems = stems[: args.limit]

    # index is compact: one task-id table per benchmark, groups reference ids by table position, outcomes are '0'/'1' strings
    bench_ids, bench_pos = {}, {}
    groups, n_frag = [], 0
    for i, (stem, meta) in enumerate(stems):
        with open(os.path.join(RESULTS, stem + ".pkl"), "rb") as f:
            recs = pickle.load(f)
        if isinstance(recs, dict):
            recs = list(recs.values())
        ids = bench_ids.setdefault(meta["bench"], [])
        pos = bench_pos.setdefault(meta["bench"], {})
        t, ok = [], []
        for idx, rec in enumerate(recs):
            frag = render_fragment(rec, meta)
            with open(os.path.join(FRAG_DIR, f"{stem}__{idx:03d}.html"), "w", encoding="utf-8") as f:
                f.write(frag)
            tid = str(rec.get("task_id"))
            if tid not in pos:
                pos[tid] = len(ids)
                ids.append(tid)
            t.append(pos[tid])
            ok.append("1" if rec.get("success") else "0")
            n_frag += 1
        groups.append({"stem": stem, "t": t, "ok": "".join(ok), **meta})
        print(f"[{i+1}/{len(stems)}] {stem}: {len(t)} tasks", flush=True)

    index = json.dumps({"bench_ids": bench_ids, "groups": groups},
                       separators=(",", ":"), ensure_ascii=False)
    with open(SHELL, "w", encoding="utf-8") as f:
        f.write(SHELL_TEMPLATE.replace("__INDEX__", index.replace("</", "<\\/")))
    print(f"wrote {SHELL} ({os.path.getsize(SHELL)/1e6:.2f} MB index) and {n_frag} fragments in {FRAG_DIR}")


if __name__ == "__main__":
    main()
