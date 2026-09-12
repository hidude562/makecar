#!/usr/bin/env python3
"""Summarise a stream-json agent log: tool calls and assistant text, most recent last.
Usage: scripts/agent_tail.py logs/<name>.jsonl [-n 30]"""
import json, sys
path = sys.argv[1]
n = int(sys.argv[sys.argv.index("-n") + 1]) if "-n" in sys.argv else 30
events = []
for line in open(path, errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        continue
    t = ev.get("type")
    if t == "assistant":
        for blk in ev.get("message", {}).get("content", []):
            if blk.get("type") == "text" and blk["text"].strip():
                events.append("ASSISTANT: " + blk["text"].strip().replace("\n", " ")[:300])
            elif blk.get("type") == "tool_use":
                inp = blk.get("input", {})
                desc = inp.get("description") or inp.get("command") or inp.get("file_path") or inp.get("pattern") or json.dumps(inp)[:120]
                events.append(f"TOOL {blk.get('name')}: {str(desc)[:160]}")
    elif t == "result":
        events.append(f"RESULT ({ev.get('subtype')}, {ev.get('num_turns')} turns, ${ev.get('total_cost_usd', 0):.2f}, {ev.get('duration_ms', 0)//1000}s): " + str(ev.get('result', ''))[:600])
print(f"{len(events)} events; last {n}:")
for e in events[-n:]:
    print(" ", e)
