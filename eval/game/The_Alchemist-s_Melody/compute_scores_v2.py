#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_scores_v15.py
========================================================
Calculate scores strictly according to the formula table below, **without any clipping/scaling/mapping**:

| Code | Formula (as ratio of total steps)                     | Range |
|------|-------------------------------------------------------|-------|
| A    | Hit rate × 30                                         | 0–30  |
| B    | 30 (if S ≤ n); otherwise 30 – excess steps (≥ 0)     | 0–30  |
| C    | (Σ correct streak lengths / S) × 10                  | 0–10  |
| D    | 10 – (Σ wrong streak lengths / S) × 10               | 0–10  |
| E    | 15 – (Σ same-color wrong lengths / S) × 15           | 0–15  |
| F    | (Color change count / (S – 1)) × 5                   | 0–5   |

Total Score: **Score = A + B + C + D + E + F**  
(Theoretical range exactly 0–100, no truncation applied)
"""

import os, re, json, glob, argparse
from collections import defaultdict
from typing import Dict, List, Any

try:
    from prettytable import PrettyTable          # 可选：pip install prettytable
except ImportError:
    PrettyTable = None

# -------------------------------------------------- 子分与总分
def compute_subscores(stats: Dict[str, Any]) -> Dict[str, float]:
    """根据统计量计算 A–F 及总分（不做 clip）"""
    n, S = stats["n"], stats["S"]
    if S == 0:
        return {k: 0.0 for k in ("A","B","C","D","E","F","Score")}

    hit_rate        = stats["hits"] / S
    corr_ratio      = sum(stats["corr_streaks"])      / S
    wrong_ratio     = sum(stats["wrong_streaks"])     / S
    rep_wrong_ratio = sum(stats["rep_wrong_streaks"]) / S
    div_ratio       = stats["color_changes"] / (S - 1) if S > 1 else 0.0

    A = 30 * hit_rate
    B = max(0, 30 - max(0, S - n))          # 超 1 步扣 1 分，最低 0
    C = 10 * corr_ratio
    D = max(0, 10 - 10 * wrong_ratio)       # 连错越多分越低
    E = max(0, 15 - 15 * rep_wrong_ratio)   # 同色错越多分越低
    F = 5 * div_ratio

    score = A + B + C + D + E + F
    return dict(A=A, B=B, C=C, D=D, E=E, F=F, Score=score)

# -------------------------------------------------- 单 episode 解析
def analyze_episode(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算一个 episode 的所有统计量与分数"""
    n = states[-1].get("sequence_length", 5)
    st = dict(
        n=n, S=len(states),
        hits=0, color_changes=0,
        corr_streaks=[], wrong_streaks=[], rep_wrong_streaks=[]
    )

    last_color = None
    cur_ok = None
    cur_len = rep_len = 0

    for idx, s in enumerate(states, 1):
        ok = bool(s["currently_in_correct_sequence"])
        col = s["last_clicked_color"]

        # 颜色变化
        if last_color is not None and col != last_color:
            st["color_changes"] += 1
        last_color = col

        # 命中统计
        if ok:
            st["hits"] += 1

        # streak 统计
        if cur_ok is None:
            cur_ok, cur_len = ok, 1
            rep_len = 1 if not ok else 0
        elif ok == cur_ok:
            cur_len += 1
            if not ok:                             # 连续错误
                prev = states[idx - 2]["last_clicked_color"] if idx > 1 else None
                if col == prev:
                    rep_len += 1                  # 同色错误串
                else:
                    if rep_len >= 2:
                        st["rep_wrong_streaks"].append(rep_len)
                    rep_len = 1
        else:                                      # streak 翻转
            if cur_ok:
                st["corr_streaks"].append(cur_len)
            else:
                st["wrong_streaks"].append(cur_len)
                if rep_len >= 2:
                    st["rep_wrong_streaks"].append(rep_len)
            cur_ok, cur_len = ok, 1
            rep_len = 1 if not ok else 0

    # 尾段处理
    if cur_ok:
        st["corr_streaks"].append(cur_len)
    else:
        st["wrong_streaks"].append(cur_len)
        if rep_len >= 2:
            st["rep_wrong_streaks"].append(rep_len)

    subs = compute_subscores(st)
    st.update({k: round(v, 2) for k, v in subs.items()})  # 改为2位小数
    return st

# -------------------------------------------------- 批量读取
RE = re.compile(r"game_state_ep(\d+)_step(\d+)\.json")
def load_episodes(root: str) -> Dict[int, List[Dict[str, Any]]]:
    eps = defaultdict(list)
    for fp in glob.glob(os.path.join(root, "game_state_ep*_step*.json")):
        m = RE.search(os.path.basename(fp));  
        if not m: continue
        ep_id, step_id = map(int, m.groups())
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_step_id"] = step_id
        eps[ep_id].append(data)

    for ep in eps:
        eps[ep].sort(key=lambda d: d["_step_id"])
    return eps

# -------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="包含 game_state_*.json 的目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每回合明细")
    args = parser.parse_args()

    episodes = load_episodes(args.root)
    if not episodes:
        print("❌ 未找到 game_state 文件"); return

    out_dir = os.path.join(args.root, "scores")
    os.makedirs(out_dir, exist_ok=True)

    table = PrettyTable() if PrettyTable else None
    if table:
        table.field_names = ["Ep", "Score", "A", "B", "C", "D", "E", "F"]

    summary = {}
    for ep_id, states in sorted(episodes.items()):
        res = analyze_episode(states)
        summary[ep_id] = res["Score"]

        with open(os.path.join(out_dir, f"episode{ep_id}.details.json"),
                  "w", encoding="utf-8") as fw:
            json.dump(res, fw, ensure_ascii=False, indent=2)

        if table:
            table.add_row([ep_id, res["Score"], res["A"], res["B"],
                           res["C"], res["D"], res["E"], res["F"]])
        elif args.verbose:
            print(json.dumps(res, indent=2, ensure_ascii=False))

    if table:
        print(table)

    with open(os.path.join(out_dir, "summary.json"),
              "w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)

    print("✅ 评分完成 (v15) – 直接使用公式，总分可高于 100 或低于 0")

if __name__ == "__main__":
    main()
