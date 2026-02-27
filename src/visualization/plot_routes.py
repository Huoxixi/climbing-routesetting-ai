import argparse
import json
import matplotlib
matplotlib.use('Agg')  # 强制静默模式，永不死锁
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from pathlib import Path
from src.env.board import Board
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="outputs/figures/action_generated_routes.jsonl")
    ap.add_argument("--out", default="outputs/figures/final_routes")
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--cols", type=int, default=11)
    args = ap.parse_args()

    # 路径清洗：剥除引号和换行符
    clean_file = args.file.strip('"\' \n\r')
    clean_out = args.out.strip('"\' \n\r')

    inp = Path(clean_file)
    if not inp.exists(): return print(f"File not found: {inp}")
    out_dir = Path(clean_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    board = Board()
    
    # 高级暗黑配色
    BG_COLOR = '#1e1e1e'
    DOT_COLOR = '#444444'
    HL_CIRCLE = '#444444'
    CYAN = '#00ffff'
    MAGENTA = '#ff00ff'
    WHITE = '#ffffff'
    
    with inp.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            
            raw_id = str(rec.get("id", "unk")).strip()
            id_str = re.sub(r'[\\/*?:"<>|\n\r]', "", raw_id)
            grade = rec.get("grade", 3)
            
            base_holds = rec.get("base_holds", [])
            finish_holds = rec.get("finish_holds", [])
            betamove = rec.get("seq_betamove", [])
            action_seq = rec.get("action_seq", [])
            
            fig, ax = plt.subplots(figsize=(7, 10), facecolor=BG_COLOR)
            ax.set_facecolor(BG_COLOR)
            ax.set_aspect('equal')
            ax.set_xlim(-1, args.cols)
            ax.set_ylim(-1, args.rows)
            ax.autoscale(False)
            ax.axis('off')

            bg_x, bg_y = [], []
            for r in range(args.rows):
                for c in range(args.cols):
                    bg_x.append(c)
                    bg_y.append(r)
            ax.scatter(bg_x, bg_y, s=120, color=DOT_COLOR, zorder=1)

            # 这里的 hold_to_num 已经把 betamove 里的点按顺序编好了 1, 2, 3...
            hold_to_num = {hid: i + 1 for i, hid in enumerate(betamove)}
            hold_to_hand = {}
            for act in action_seq:
                if "_H" in act:
                    try:
                        hid = int(act.split("_H")[-1])
                        hold_to_hand[hid] = 'lh' if "LH" in act else 'rh'
                    except: pass

            lh_sequence = [hid for hid in betamove if hold_to_hand.get(hid) == 'lh']
            rh_sequence = [hid for hid in betamove if hold_to_hand.get(hid) == 'rh']

            for i in range(len(lh_sequence) - 1):
                r1, c1 = board.from_id(lh_sequence[i])
                r2, c2 = board.from_id(lh_sequence[i+1])
                ax.plot([c1, c2], [r1, r2], color=CYAN, linewidth=3, zorder=5)

            for i in range(len(rh_sequence) - 1):
                r1, c1 = board.from_id(rh_sequence[i])
                r2, c2 = board.from_id(rh_sequence[i+1])
                ax.plot([c1, c2], [r1, r2], color=MAGENTA, linewidth=3, zorder=5)

            for hid in set(base_holds + finish_holds + betamove):
                r, c = board.from_id(hid)
                ax.add_patch(plt.Circle((c, r), 0.5, color=HL_CIRCLE, alpha=0.5, zorder=4))
                
                fill_col = CYAN if hold_to_hand.get(hid) == 'lh' else MAGENTA
                num = hold_to_num.get(hid, 0)
                
                # 🚨 极其清爽的逻辑：B点画白，F点画白，其他的直接显示序号！
                if hid in base_holds:
                    label, fill_col = 'B', WHITE
                elif hid in finish_holds:
                    label, fill_col = 'F', WHITE
                else:
                    label = str(num) if num > 0 else ""
                
                font_color = 'black' if fill_col in [CYAN, WHITE] else WHITE
                
                ax.add_patch(plt.Circle((c, r), 0.35, color=fill_col, ec=WHITE, lw=1.5, zorder=10))
                if label:
                    ax.text(c, r, label, color=font_color, fontsize=12, fontweight='bold', ha='center', va='center', zorder=20)

            ax.text(args.cols/2 - 0.5, args.rows - 0.2, f"Biomechanics Trace | V{grade}", color=WHITE, fontsize=16, fontweight='bold', ha='center', va='center')
            lh_legend = mlines.Line2D([], [], color=CYAN, marker='o', markersize=10, markerfacecolor=CYAN, markeredgecolor=WHITE, label='Left Hand (L)')
            rh_legend = mlines.Line2D([], [], color=MAGENTA, marker='o', markersize=10, markerfacecolor=MAGENTA, markeredgecolor=WHITE, label='Right Hand (R)')
            ax.legend(handles=[lh_legend, rh_legend], loc='upper left', frameon=False, labelcolor=WHITE, fontsize=12)

            # 强转绝对路径
            save_path = str((out_dir / f"{id_str}.png").absolute())
            fig.savefig(save_path, dpi=150, facecolor=BG_COLOR)
            plt.close(fig)

    print("✅ 画图完毕，起手已变更为序号 1、2！")

if __name__ == "__main__":
    main()