#!/usr/bin/env python3
import argparse
import html
import json
import os
from typing import List, Dict, Any, Optional
import re

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                # Skip malformed lines but keep going
                print(f"[warn] skip bad json line: {e}")
    return records


def color_for_score(score: float) -> str:
    # Map 0..1 to red(0) -> yellow(0.5) -> green(1)
    # Use HSL hue: 0 (red) to 120 (green)
    hue = int(max(0.0, min(1.0, score)) * 120)
    return f"hsl({hue}, 70%, 90%)"


def badge_html(idx: int, score: float, probs=None, tool_name: Optional[str] = None) -> str:
    bg = color_for_score(score)
    title = f"marker#{idx} score={score:.3f}"
    if probs and isinstance(probs, (list, tuple)) and len(probs) == 2:
        title += f"\nprobs=[{probs[0]:.3f}, {probs[1]:.3f}]"
    label = f"marker#{idx}"
    if tool_name:
        label += f" · {html.escape(tool_name)}"
    return (
        f"<span class=\"marker-badge\" style=\"background:{bg};\" title=\"{html.escape(title)}\">"
        f"<b>{label}</b> · score={score:.3f}"
        "</span>"
    )


def _escape_with_question_highlight(text: str) -> str:
    # Highlight 'Question: ...' (or 'Question：...') until newline or '<'
    pattern = re.compile(r"Question[:：]\s*([^\n\r<]+)")
    out: List[str] = []
    last = 0
    for m in pattern.finditer(text):
        start, end = m.span()
        out.append(html.escape(text[last:start]))
        full = text[start:end]
        out.append(
            f"<span class=\"question-highlight\">{html.escape(full)}</span>"
        )
        last = end
    out.append(html.escape(text[last:]))
    return "".join(out)


def _extract_question(text: str) -> Optional[str]:
    m = re.search(r"Question[:：]\s*([^\n\r<]+)", text)
    if m:
        return m.group(1).strip()
    return None


def record_to_html(rec: Dict[str, Any], marker_token: str, gt_text: Optional[str] = None) -> str:
    text = rec.get("text", "")
    markers = rec.get("markers", []) or []

    # Replace marker tokens in order with colored badges
    out_parts: List[str] = []
    pos = 0
    count = 0
    while True:
        found = text.find(marker_token, pos)
        if found == -1:
            out_parts.append(html.escape(text[pos:]))
            break
        # pre-chunk
        out_parts.append(_escape_with_question_highlight(text[pos:found]))
        # badge for this marker occurrence
        if count < len(markers):
            m = markers[count]
            score = float(m.get("score_pos", 0.0))
            probs = m.get("probs", None)
            # Try to infer tool name from nearest preceding <tool_call> ... {"name": "..."}
            lookback_start = max(0, found - 1200)
            context = text[lookback_start:found]
            tool_name = None
            # Ensure we grab the last <tool_call> block before marker
            tc_pos = context.rfind("<tool_call>")
            if tc_pos != -1:
                # From this <tool_call> to marker, extract name
                snippet = context[tc_pos:]
                m_name = re.search(r'"name"\s*:\s*"([^"]+)"', snippet)
                if m_name:
                    tool_name = m_name.group(1)
            out_parts.append(badge_html(count, score, probs, tool_name))
        else:
            out_parts.append('<span class="marker-badge missing">marker</span>')
        pos = found + len(marker_token)
        count += 1

    meta = {
        "global_index": rec.get("global_index"),
        "marker_count": rec.get("marker_count"),
        "source_file": rec.get("source_file"),
    }

    html_block = (
        "<div class=\"sample\">"
        f"<div class=\"meta\"><code>idx={meta['global_index']}</code> · "
        f"<code>markers={meta['marker_count']}</code> · "
        f"<code>source={html.escape(str(meta['source_file']))}</code></div>"
        f"<pre class=\"text\">{''.join(out_parts)}</pre>"
    )
    if gt_text is not None:
        safe = html.escape(str(gt_text))
        html_block += f"<div class=\"gt\"><b>Ground Truth</b>: <code>{safe}</code></div>"
    html_block += "</div>"
    return html_block


def build_html(records: List[Dict[str, Any]], marker_token: str, max_samples: int, gt_by_idx: Optional[Dict[int, Any]] = None) -> str:
    head = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Marker PRM Insight</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }}
    .legend {{ margin-bottom: 12px; }}
    .marker-badge {{ padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.1); font-size: 12px; }}
    .marker-badge.missing {{ background: #eee; color: #666; }}
    .sample {{ margin: 14px 0; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }}
    .meta {{ color: #4b5563; margin-bottom: 8px; }}
    .text {{ white-space: pre-wrap; word-break: break-word; background: #fff; padding: 10px; border-radius: 6px; border: 1px solid #eee; }}
    .question-highlight {{ background: #fff3cd; border-radius: 3px; padding: 0 2px; }}
    code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>Marker PRM Insight</h2>
  <div class=\"legend\">
    Replacing occurrences of <code>{html.escape(marker_token)}</code> with color-coded badges.
    Color encodes score (red → green for 0.0 → 1.0). Hover to see details.
  </div>
"""

    body = []
    for i, rec in enumerate(records[:max_samples]):
        gi = int(rec.get("global_index", i))
        gt_text = None
        if gt_by_idx is not None:
            gt_text = gt_by_idx.get(gi)
        body.append(record_to_html(rec, marker_token, gt_text))

    tail = """
</body>
</html>
"""
    return head + "\n".join(body) + tail


def tty_summary(records: List[Dict[str, Any]], marker_token: str, max_samples: int, gt_by_idx: Optional[Dict[int, Any]] = None) -> None:
    print(f"Loaded {len(records)} eval records. Showing first {min(len(records), max_samples)}.")
    for rec in records[:max_samples]:
        gi = rec.get("global_index")
        mks = rec.get("markers", []) or []
        scores = [float(m.get("score_pos", 0.0)) for m in mks]
        # Build a compact bar with threshold coloring
        def seg(s: float) -> str:
            if s >= 0.8:
                color = "\x1b[32m"  # green
            elif s >= 0.5:
                color = "\x1b[33m"  # yellow
            elif s >= 0.2:
                color = "\x1b[35m"  # magenta
            else:
                color = "\x1b[31m"  # red
            return f"{color}{s:0.3f}\x1b[0m"

        bar = " | ".join(seg(s) for s in scores) if scores else "(no markers)"
        suffix = ""
        if gt_by_idx is not None:
            gt = gt_by_idx.get(int(gi))
            if gt is not None:
                suffix = f"  GT={gt}"
        print(f"idx={gi} markers={len(mks)} -> {bar}{suffix}")

        # Show a short context around the first marker occurrence in text
        text = rec.get("text", "")
        q = _extract_question(text)
        if q:
            print(f"  Q: {q}")
        pos = text.find(marker_token)
        if pos != -1:
            left = max(0, pos - 80)
            right = min(len(text), pos + len(marker_token) + 80)
            snippet = text[left:pos] + "[MARKER]" + text[pos + len(marker_token):right]
            print("  …" + snippet.replace("\n", " ") + "…")


def _extract_ground_truth_list(df) -> List[Any]:
    # Try reward_model.ground_truth first
    if df is None:
        return []
    try:
        if "reward_model" in df.columns:
            ser = df["reward_model"]
            def get_gt(v):
                try:
                    if isinstance(v, str):
                        try:
                            obj = json.loads(v)
                            if isinstance(obj, dict) and "ground_truth" in obj:
                                return obj["ground_truth"]
                            return v
                        except Exception:
                            return v
                    if isinstance(v, dict):
                        return v.get("ground_truth")
                    # pandas Series or object with get
                    if hasattr(v, "get"):
                        return v.get("ground_truth")
                    return v
                except Exception:
                    return None
            return ser.apply(get_gt).tolist()
        if "ground_truth" in df.columns:
            return df["ground_truth"].tolist()
    except Exception:
        pass
    # Fallback: nothing
    return [None] * len(df)


def _load_gt_by_index(parquet_path: str) -> Optional[Dict[int, Any]]:
    if not parquet_path:
        return None
    if pd is None:
        print("[warn] pandas not installed; skip loading ground_truth from parquet")
        return None
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[warn] failed to read parquet: {parquet_path}: {e}")
        return None
    gts = _extract_ground_truth_list(df)
    return {i: gts[i] for i in range(len(gts))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PRM marker scores with inline badges.")
    parser.add_argument(
        "--eval_path",
        type=str,
        default="/nfsdata/fanshengda/verl/near_miss_pair_PRM/output/prm_anchor_validation_1128_ckpt3500.jsonl",
        help="Path to eval JSONL from eval_marker_prm.py",
    )
    parser.add_argument("--out", type=str, default="/nfsdata/fanshengda/verl/near_miss_pair_PRM/output/test.html", help="Optional output HTML path. Defaults to <eval_path>_insight.html")
    parser.add_argument("--marker_token", type=str, default="<extra_0>", help="Marker token string to replace")
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="/nfsdata/fanshengda/verl/input_data/near_miss_prm/anchor_validation_1116.parquet",
        help="Parquet path that contains reward_model.ground_truth",
    )
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to render in HTML/TTY")
    args = parser.parse_args()

    records = read_jsonl(args.eval_path)
    if not records:
        print(f"No records found in {args.eval_path}")
        return

    # Load ground truth list by index
    gt_by_idx: Optional[Dict[int, Any]] = None
    parquet_path = args.parquet_path
    if not parquet_path:
        # Try infer from first record
        parquet_path = str(records[0].get("source_file") or "")
    if parquet_path:
        gt_by_idx = _load_gt_by_index(parquet_path)

    # TTY summary
    tty_summary(records, args.marker_token, args.max_samples, gt_by_idx)

    # HTML output
    out_path = args.out
    if not out_path:
        base = os.path.basename(args.eval_path)
        out_path = os.path.join(os.path.dirname(args.eval_path), base + ".insight.html")

    html_doc = build_html(records, args.marker_token, args.max_samples, gt_by_idx)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Wrote HTML report -> {out_path}")


if __name__ == "__main__":
    main()
