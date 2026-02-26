import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from pathlib import Path
from src.env.board import Board

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="outputs/figures/action_generated_routes.jsonl")
    ap.add_argument("--out", default="outputs/figures/final_routes")
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--cols", type=int, default=11)
    args = ap.parse_args()

    inp = Path(args.file)
    if not inp.exists(): return print(f"File not found: {args.file}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    board = Board()
    
    # === 你的黄金时代高级暗黑配色 ===
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
            id_str, grade = rec.get("id", "unk"), rec.get("grade", 3)
            betamove = rec.get("seq_betamove", [])
            action_seq = rec.get("action_seq", [])
            
            fig, ax = plt.subplots(figsize=(7, 10), facecolor=BG_COLOR)
            ax.set_facecolor(BG_COLOR)
            ax.set_aspect('equal')
            ax.set_xlim(-1, args.cols)
            ax.set_ylim(-1, args.rows)
            ax.axis('off')

            # 画底部的深灰色岩点阵列
            for r in range(args.rows):
                for c in range(args.cols):
                    ax.add_patch(plt.Circle((c, r), 0.15, color=DOT_COLOR, zorder=1))

            hold_to_num = {}
            for i, hid in enumerate(betamove):
                if hid not in hold_to_num: hold_to_num[hid] = i + 1

            hold_to_hand = {}
            for act in action_seq:
                if "_H" in act:
                    try:
                        hid = int(act.split("_H")[-1])
                        hold_to_hand[hid] = 'lh' if "LH" in act else 'rh'
                    except: pass

            # ==========================================================
            # 🚨 绝对核心修复：将点分为左手和右手，分别画两条完全独立的线
            # ==========================================================
            lh_holds = [hid for hid in betamove if hold_to_hand.get(hid) == 'lh']
            rh_holds = [hid for hid in betamove if hold_to_hand.get(hid) == 'rh']

            # 1. 画青色左手独立连线 (左手只连左手)
            for i in range(len(lh_holds) - 1):
                r1, c1 = board.from_id(lh_holds[i])
                r2, c2 = board.from_id(lh_holds[i+1])
                ax.plot([c1, c2], [r1, r2], color=CYAN, linewidth=3, zorder=5)

            # 2. 画洋红右手独立连线 (右手只连右手)
            for i in range(len(rh_holds) - 1):
                r1, c1 = board.from_id(rh_holds[i])
                r2, c2 = board.from_id(rh_holds[i+1])
                ax.plot([c1, c2], [r1, r2], color=MAGENTA, linewidth=3, zorder=5)

            # === 画点、光晕、数字和 'S' 标识 ===
            for hid in set(betamove):
                r, c = board.from_id(hid)
                # 高级感来源：半透明发光底座
                ax.add_patch(plt.Circle((c, r), 0.5, color=HL_CIRCLE, alpha=0.5, zorder=4))
                
                fill_col = CYAN if hold_to_hand.get(hid) == 'lh' else MAGENTA
                
                num = hold_to_num.get(hid, 0)
                # 黄金时代逻辑：前两手标定为 S，后面依次排数字 1, 2, 3...
                if num in [1, 2]:
                    label = 'S'
                else:
                    label = str(num - 2)
                
                # 黑色字在Cyan上，白色字在Magenta上
                font_color = 'black' if fill_col == CYAN else WHITE
                
                ax.add_patch(plt.Circle((c, r), 0.35, color=fill_col, ec=WHITE, lw=1.5, zorder=10))
                ax.text(c, r, label, color=font_color, fontsize=12, fontweight='bold', ha='center', va='center', zorder=20)

            # 顶部标题和图例
            ax.text(args.cols/2 - 0.5, args.rows - 0.2, f"Biomechanics Trace | V{grade}", color=WHITE, fontsize=16, fontweight='bold', ha='center', va='center')
            lh_legend = mlines.Line2D([], [], color=CYAN, marker='o', markersize=10, markerfacecolor=CYAN, markeredgecolor=WHITE, label='Left Hand (L)')
            rh_legend = mlines.Line2D([], [], color=MAGENTA, marker='o', markersize=10, markerfacecolor=MAGENTA, markeredgecolor=WHITE, label='Right Hand (R)')
            ax.legend(handles=[lh_legend, rh_legend], loc='upper left', frameon=False, labelcolor=WHITE, fontsize=12)

            fig.savefig(out_dir / f"{id_str}.png", dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
            plt.close(fig)

    print("✅ 两条独立轨迹完美分离！高级暗黑风画图完毕！")

if __name__ == "__main__":
    main()