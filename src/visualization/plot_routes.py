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
    
    # === 完美复刻你的暗黑高级配色 ===
    BG_COLOR = '#1e1e1e'
    DOT_COLOR = '#444444'
    HL_CIRCLE = '#333333'
    CYAN = '#00ffff'
    MAGENTA = '#ff00ff'
    WHITE = '#ffffff'
    
    with inp.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            id_str, grade = rec.get("id", "unk"), rec.get("grade", 3)
            base_holds = rec.get("base_holds", [])
            finish_holds = rec.get("finish_holds", [])
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

            # 提取序号和左右手
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

            # === 灵魂修复 1：画青色/洋红连线！(绝不能少) ===
            for i in range(len(betamove) - 1):
                r1, c1 = board.from_id(betamove[i])
                r2, c2 = board.from_id(betamove[i+1])
                hand_color = CYAN if hold_to_hand.get(betamove[i+1]) == 'lh' else MAGENTA
                ax.plot([c1, c2], [r1, r2], color=hand_color, linewidth=3, zorder=5)

            all_hids = set(base_holds + finish_holds + betamove)
            for hid in all_hids:
                r, c = board.from_id(hid)
                # === 灵魂修复 2：半透明高级光晕 ===
                ax.add_patch(plt.Circle((c, r), 0.45, color=HL_CIRCLE, alpha=0.6, zorder=4))
                
                label, edge_col, fill_col = "", WHITE, WHITE
                
                # 优先级判断：B点/F点优先显示为白色底，数字点显示为彩色底
                if hid in base_holds:
                    label, fill_col, edge_col = 'B', WHITE, MAGENTA 
                elif hid in finish_holds:
                    label, fill_col, edge_col = 'F', WHITE, CYAN    
                elif hid in hold_to_num:
                    label = str(hold_to_num[hid])
                    fill_col = CYAN if hold_to_hand.get(hid) == 'lh' else MAGENTA
                    edge_col = WHITE
                
                # 画内圈圆点和数字
                ax.add_patch(plt.Circle((c, r), 0.25, color=fill_col, ec=edge_col, lw=1.5, zorder=10))
                ax.text(c, r, label, color='black', 
                        fontsize=11, fontweight='bold', ha='center', va='center', zorder=20)

            # 顶部标题和图例 (白色字体)
            ax.text(args.cols/2 - 0.5, args.rows - 0.2, f"Biomechanics Trace | V{grade}", 
                    color=WHITE, fontsize=16, fontweight='bold', ha='center', va='center')
            
            # 自定义图例
            lh_legend = mlines.Line2D([], [], color=CYAN, marker='o', markersize=10, markerfacecolor=CYAN, markeredgecolor=WHITE, label='Left Hand (L)')
            rh_legend = mlines.Line2D([], [], color=MAGENTA, marker='o', markersize=10, markerfacecolor=MAGENTA, markeredgecolor=WHITE, label='Right Hand (R)')
            ax.legend(handles=[lh_legend, rh_legend], loc='upper left', frameon=False, labelcolor=WHITE, fontsize=12)

            fig.savefig(out_dir / f"{id_str}.png", dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
            plt.close(fig)

    print("✅ 颜值恢复！高级暗黑风画图完毕！")

if __name__ == "__main__":
    main()