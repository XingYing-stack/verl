#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import html
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------------------- Parsing --------------------

RE_BLOCK_Q  = re.compile(r"\*\*Question:\*\*\s*(.*?)\n\s*---", re.S)
RE_BLOCK_T1 = re.compile(r"\*\*trajectory_1:\*\*\s*```json\s*(.*?)\s*```\s*", re.S)
RE_BLOCK_T2 = re.compile(r"\*\*trajectory_2:\*\*\s*```json\s*(.*?)\s*```\s*", re.S)
RE_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)

def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def parse_task_from_input_prompt(input_str: str) -> Dict[str, Any]:
    """Extract question + trajectories from your long prompt string."""
    q = None
    m = RE_BLOCK_Q.search(input_str)
    if m:
        q = m.group(1).strip()

    t1 = None
    m = RE_BLOCK_T1.search(input_str)
    if m:
        t1 = safe_json_loads(m.group(1).strip())

    t2 = None
    m = RE_BLOCK_T2.search(input_str)
    if m:
        t2 = safe_json_loads(m.group(1).strip())

    return {"question": q, "trajectory_1": t1, "trajectory_2": t2}

def parse_judge_json(output_str: str) -> Optional[Dict[str, Any]]:
    """Extract judge JSON from ```json ...``` inside model output."""
    m = RE_JSON_FENCE.search(output_str)
    if not m:
        return None
    return safe_json_loads(m.group(1))

def extract_thought_block(output_str: str) -> str:
    """
    Extract thought text from the beginning of output up to </think> (inclusive).
    If </think> not found, return empty string.
    """
    if not output_str:
        return ""
    end_tag = "</think>"
    end_idx = output_str.find(end_tag)
    if end_idx == -1:
        return ""
    return output_str[: end_idx + len(end_tag)]

def get_steps(traj: Any) -> List[Dict[str, Any]]:
    """
    Support:
    - list[{step_id, content}]  (your example)
    - dict{"0": "...", "1": "..."} (your spec)
    """
    if traj is None:
        return []
    if isinstance(traj, list):
        if all(isinstance(x, dict) and "step_id" in x for x in traj):
            return sorted(traj, key=lambda x: int(x["step_id"]))
        return [{"step_id": i, "content": str(x)} for i, x in enumerate(traj)]
    if isinstance(traj, dict):
        items = []
        for k, v in traj.items():
            try:
                sid = int(k)
            except Exception:
                continue
            items.append({"step_id": sid, "content": str(v)})
        return sorted(items, key=lambda x: x["step_id"])
    return []

def step_flags(step_text: str) -> List[str]:
    flags = []
    if "<tool_call>" in step_text:
        flags.append("tool_call")
    if "<tool_response>" in step_text:
        flags.append("tool_response")
    if "<answer>" in step_text:
        flags.append("answer")
    return flags

def eval_map(judge: Dict[str, Any], which: str) -> Dict[int, Dict[str, Any]]:
    """
    which in {"trajectory_1","trajectory_2"}
    returns {step_id: {step_id, reason, helps}}
    """
    se = (judge or {}).get("step_evaluations", {})
    arr = se.get(which, []) if isinstance(se, dict) else []
    m = {}
    for e in arr:
        if not isinstance(e, dict) or "step_id" not in e:
            continue
        try:
            sid = int(e["step_id"])
        except Exception:
            continue
        m[sid] = e
    return m

# -------------------- HTML Rendering --------------------

def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))

def helps_class(helps: Any) -> str:
    if helps is True:
        return "good"
    if helps is False:
        return "bad"
    return "unk"

def helps_badge(helps: Any) -> str:
    if helps is True:
        return "✅"
    if helps is False:
        return "❌"
    return "？"

def build_heatmap_row(traj_name: str, steps: List[Dict[str, Any]], emap: Dict[int, Dict[str, Any]]) -> str:
    cells = []
    for s in steps:
        sid = int(s.get("step_id", -1))
        e = emap.get(sid, {})
        helps = e.get("helps", None)
        cls = helps_class(helps)
        badge = helps_badge(helps)
        reason = (e.get("reason", "") if isinstance(e, dict) else "")
        title = f"{traj_name} step {sid} | helps={helps} | {reason}"
        cells.append(f'<div class="hm-cell {cls}" title="{esc(title)}">{esc(badge)}<span class="hm-id">{sid}</span></div>')
    return f'<div class="hm-row"><div class="hm-label">{esc(traj_name)}</div><div class="hm-cells">{"".join(cells)}</div></div>'

def build_steps_table(traj_key: str, steps: List[Dict[str, Any]], emap: Dict[int, Dict[str, Any]]) -> str:
    rows = []
    for s in steps:
        sid = int(s.get("step_id", -1))
        content = s.get("content", "") or ""
        flags = ", ".join(step_flags(content))
        e = emap.get(sid, {})
        helps = e.get("helps", None)
        reason = e.get("reason", "") if isinstance(e, dict) else ""
        cls = helps_class(helps)
        badge = helps_badge(helps)

        rows.append(f"""
        <div class="step {cls}">
          <div class="bar"></div>
          <div class="step-main">
            <div class="step-head">
              <div class="step-left">
                <span class="badge">{esc(badge)}</span>
                <span class="mono step-id">{esc(traj_key)}:{sid}</span>
                <span class="chip">{esc(flags) if flags else "—"}</span>
              </div>
              <button class="btn" onclick="toggle('c_{esc(traj_key)}_{sid}')">Show/Hide content</button>
            </div>
            <div class="reason"><span class="mono">reason:</span> {esc(reason) if reason else "<span class='muted'>—</span>"}</div>
            <pre class="content" id="c_{esc(traj_key)}_{sid}">{esc(content)}</pre>
          </div>
        </div>
        """)
    return "\n".join(rows)

def write_html(
    out_path: Path,
    record: Dict[str, Any],
    question: str,
    t1_steps: List[Dict[str, Any]],
    t2_steps: List[Dict[str, Any]],
    judge: Dict[str, Any],
    thought_text: str,
):
    e1 = eval_map(judge, "trajectory_1")
    e2 = eval_map(judge, "trajectory_2")
    fc = (judge or {}).get("final_comparison", {}) if judge else {}

    meta_items = {
        "score": record.get("score"),
        "reward": record.get("reward"),
        "pc_json_found": record.get("pc_json_found"),
        "pc_step_len_t1": record.get("pc_step_len_t1"),
        "pc_step_len_t2": record.get("pc_step_len_t2"),
        "gts": record.get("gts"),
    }

    html_str = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Judge Help Visualization</title>
<style>
  :root {{
    --fg: #111;
    --muted: #6b7280;
    --border: #e5e7eb;
    --card: #fff;
    --bg: #f6f7fb;
    --good: #16a34a;
    --bad: #dc2626;
    --unk: #d97706;
    --goodBg: #ecfdf5;
    --badBg: #fff1f2;
    --unkBg: #fffbeb;
  }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif;
  }}
  .wrap {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
  h1 {{ font-size: 20px; margin: 0 0 8px 0; }}
  h2 {{ font-size: 16px; margin: 0 0 10px 0; }}
  .mono {{ font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; }}
  .muted {{ color: var(--muted); }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 980px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}

  /* Heatmap */
  .hm {{
    display: grid;
    gap: 8px;
  }}
  .hm-row {{
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 10px;
    align-items: center;
  }}
  .hm-label {{
    color: var(--muted);
    font-weight: 600;
  }}
  .hm-cells {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .hm-cell {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid var(--border);
    font-weight: 700;
    cursor: default;
    user-select: none;
  }}
  .hm-cell .hm-id {{
    font-weight: 600;
    color: var(--muted);
  }}
  .hm-cell.good {{ background: var(--goodBg); }}
  .hm-cell.bad  {{ background: var(--badBg); }}
  .hm-cell.unk  {{ background: var(--unkBg); }}

  /* Steps */
  .step {{
    display: grid;
    grid-template-columns: 8px 1fr;
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 10px;
    background: #fff;
  }}
  .step .bar {{ background: var(--unk); }}
  .step.good .bar {{ background: var(--good); }}
  .step.bad  .bar {{ background: var(--bad); }}
  .step.unk  .bar {{ background: var(--unk); }}
  .step-main {{ padding: 12px; }}
  .step-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
  }}
  .step-left {{
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }}
  .badge {{
    font-size: 18px;
    width: 28px;
    text-align: center;
  }}
  .step-id {{
    font-weight: 800;
  }}
  .chip {{
    font-size: 12px;
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
    white-space: nowrap;
  }}
  .btn {{
    border: 1px solid var(--border);
    background: #fff;
    padding: 6px 10px;
    border-radius: 10px;
    cursor: pointer;
    font-weight: 600;
  }}
  .btn:hover {{ background: #f3f4f6; }}
  .reason {{
    font-size: 13px;
    line-height: 1.4;
    margin-bottom: 8px;
  }}
  pre.content {{
    display: none;
    margin: 0;
    padding: 10px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: #0b1020;
    color: #e5e7eb;
    max-height: 360px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
  }}

  /* Meta table */
  .meta {{
    display: grid;
    grid-template-columns: 160px 1fr;
    gap: 8px 12px;
    font-size: 13px;
  }}
</style>
<script>
function toggle(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  if (el.style.display === 'block') el.style.display = 'none';
  else el.style.display = 'block';
}}
function showAll() {{
  document.querySelectorAll('pre.content').forEach(e => e.style.display = 'block');
}}
function hideAll() {{
  document.querySelectorAll('pre.content').forEach(e => e.style.display = 'none');
}}
</script>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Judge Help Visualization</h1>
      <div class="muted">Focus: per-step <span class="mono">helps</span> highlighting (green/red/amber).</div>
    </div>

    <div class="card">
      <h2>Question</h2>
      <div>{esc(question) if question else "<span class='muted'>(question not found)</span>"}</div>
    </div>

    <div class="card">
      <h2>Judge Thought</h2>
      <div class="muted">Extracted from <span class="mono">output</span>: beginning → <span class="mono">&lt;/think&gt;</span></div>
      <div style="margin-top:10px;">
        <button class="btn" onclick="toggle('judge_thought')">Show/Hide thought</button>
      </div>
      <pre class="content" id="judge_thought" style="display:block;">{esc(thought_text) if thought_text else "(no </think> found in output)"}</pre>
    </div>

    <div class="card">
      <h2>Final Comparison</h2>
      <div class="meta">
        <div class="muted">trajectory_1_final_correct</div><div class="mono">{esc(fc.get("trajectory_1_final_correct"))}</div>
        <div class="muted">trajectory_2_final_correct</div><div class="mono">{esc(fc.get("trajectory_2_final_correct"))}</div>
        <div class="muted">reason</div><div>{esc(fc.get("reason",""))}</div>
      </div>
    </div>

    <div class="card">
      <h2>Help Heatmap</h2>
      <div class="hm">
        {build_heatmap_row("trajectory_1", t1_steps, e1)}
        {build_heatmap_row("trajectory_2", t2_steps, e2)}
      </div>
      <div style="margin-top:10px;">
        <button class="btn" onclick="showAll()">Show all step contents</button>
        <button class="btn" onclick="hideAll()">Hide all step contents</button>
      </div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>Trajectory 1 Steps</h2>
        {build_steps_table("t1", t1_steps, e1)}
      </div>

      <div class="card">
        <h2>Trajectory 2 Steps</h2>
        {build_steps_table("t2", t2_steps, e2)}
      </div>
    </div>

    <div class="card">
      <h2>Record Meta</h2>
      <div class="meta">
        {"".join([f"<div class='muted'>{esc(k)}</div><div class='mono'>{esc(v)}</div>" for k, v in meta_items.items()])}
      </div>
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html_str, encoding="utf-8")

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default='/Users/dada/Desktop/verl/rollout_data/qwen3-4B-2507-SearchR1-1207-DrGRPO-validation/200.jsonl', help="jsonl file")
    ap.add_argument("--start", type=int, default=0, help="which record to render")
    ap.add_argument("--end", type=int, default=100, help="which record to render")

    ap.add_argument("--out", type=str, default='./output_report', help="output html path")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.path))
    if not rows:
        raise SystemExit("Empty jsonl.")

    n = len(rows)
    if not (0 <= args.start < n):
        raise SystemExit(f"--start out of range: 0..{n - 1}")
    if not (args.start < args.end <= n):
        raise SystemExit(f"--end must satisfy start < end <= {n}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.start, args.end):
        record = rows[idx]

        task = parse_task_from_input_prompt(record.get("input", ""))
        judge = parse_judge_json(record.get("output", "")) or {}

        question = task.get("question") or "(question not found)"
        t1_steps = get_steps(task.get("trajectory_1"))
        t2_steps = get_steps(task.get("trajectory_2"))

        thought_text = extract_thought_block(record.get("output", ""))

        out_path = out_dir / f"{idx}.html"

        write_html(out_path, record, question, t1_steps, t2_steps, judge, thought_text)
        print(f"Wrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()
