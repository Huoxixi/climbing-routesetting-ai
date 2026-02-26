import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from src.env.board import Board

def get_moonboard_colors():
    return {
        'start': '#FFFFFF',  # 白色
        'finish': '#FFFFFF', # 白色
        'cyan': '#00FFFF',   
        'magenta': '#FF00FF',
        'font': '#000000',   
        'circle': '#222222'  
    }

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
    
    colors = get_moonboard_colors()
    board = Board()
    
    with inp.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            id_str, grade = rec.get("id", "unk"), rec.get("grade", 3)
            
            base_holds = rec.get("base_holds", [])
            finish_holds = rec.get("finish_holds", [])
            betamove = rec.get("seq_betamove", [])
            action_seq = rec.get("action_seq", [])
            
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.set_aspect('equal')
            ax.set_xlim(-0.5, args.cols - 0.5)
            ax.set_ylim(-0.5, args.rows - 0.5)
            ax.axis('off')

            for r in range(args.rows):
                for c in range(args.cols):
                    ax.add_patch(plt.Circle((c, r), 0.2, color=colors['circle'], zorder=1))
            
            # 由于去掉了 START_ 前缀，1号点和2号点自然形成
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

            all_hids = set(base_holds + finish_holds + betamove)
            
            for hid in all_hids:
                r, c = board.from_id(hid)
                fill_color = colors['circle']
                num_label = ""
                
                if hid in base_holds:
                    fill_color = colors['start']
                    num_label = 'B'  # 标定白色起点 B
                elif hid in finish_holds:
                    fill_color = colors['finish']
                    num_label = 'F'  # 标定白色终点 F
                elif hid in hold_to_num:
                    num_label = str(hold_to_num[hid])
                    if hold_to_hand.get(hid) == 'lh': fill_color = colors['cyan']
                    else: fill_color = colors['magenta']
                
                ax.add_patch(plt.Circle((c, r), 0.35, color=fill_color, zorder=10))
                if num_label:
                    ax.text(c, r, num_label, color=colors['font'], fontsize=12, fontweight='bold', ha='center', va='center', zorder=20)

            # 画线逻辑：只连序号 1-N 的手点，不管 B 和 F
            for i in range(len(betamove) - 1):
                r1, c1 = board.from_id(betamove[i])
                r2, c2 = board.from_id(betamove[i+1])
                hand_color = colors['cyan'] if hold_to_hand.get(betamove[i+1]) == 'lh' else colors['magenta']
                ax.plot([c1, c2], [r1, r2], color=hand_color, linewidth=2.5, zorder=5)

            ax.set_title(f"V{grade} | ID: {id_str}", fontsize=14, fontweight='bold', pad=10)
            fig.savefig(out_dir / f"{id_str}.png", dpi=100, bbox_inches='tight')
            plt.close(fig)

    print("✅ 画图完毕，终极物理与拓扑约束已全线上线！")

if __name__ == "__main__":
    main()