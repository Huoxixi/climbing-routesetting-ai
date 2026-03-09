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
    
    # 高级暗黑配色 + 新手友好亮色
    BG_COLOR = '#1e1e1e'
    DOT_COLOR = '#444444'
    HL_CIRCLE = '#555555'
    CYAN = '#00e5ff'     # 左手：更明亮的青色
    MAGENTA = '#ff4081'  # 右手：更活泼的粉紫
    START_COL = '#2ecc71'# 起步：醒目的绿色
    TOP_COL = '#f1c40f'  # 完攀：胜利的金色
    WHITE = '#ffffff'
    
    with inp.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            
            raw_id = str(rec.get("id", "unk")).strip()
            id_str = re.sub(r'[\\/*?:"<>|\n\r]', "", raw_id)
            grade = rec.get("grade", 1)  # 默认降为 V1
            
            base_holds = rec.get("base_holds", [])
            finish_holds = rec.get("finish_holds", [])
            betamove = rec.get("seq_betamove", [])
            action_seq = rec.get("action_seq", [])
            
            fig, ax = plt.subplots(figsize=(8, 11), facecolor=BG_COLOR)
            ax.set_facecolor(BG_COLOR)
            ax.set_aspect('equal')
            # 扩大视野范围，为了给坐标轴留出空间
            ax.set_xlim(-1.5, args.cols + 0.5)
            ax.set_ylim(-1.5, args.rows + 1.0)
            ax.autoscale(False)
            ax.axis('off')

            # 1. 画背景岩点阵列
            bg_x, bg_y = [], []
            for r in range(args.rows):
                for c in range(args.cols):
                    bg_x.append(c)
                    bg_y.append(r)
            ax.scatter(bg_x, bg_y, s=100, color=DOT_COLOR, zorder=1)

            # 2. 画周围的坐标轴 (A-K, 1-18) 方便新手找点
            for c in range(args.cols):
                ax.text(c, -1, chr(65+c), color='#888888', fontsize=12, fontweight='bold', ha='center', va='center')
            for r in range(args.rows):
                # 调整 row 的显示，如果是从 1 到 18
                ax.text(-1, r, str(r+1) if r+1 >= 10 else f" {r+1}", color='#888888', fontsize=12, fontweight='bold', ha='center', va='center')

            # 3. 解析左右手
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

            # 4. 画连线 (加粗，体现新手路线的稳定过渡)
            for i in range(len(lh_sequence) - 1):
                r1, c1 = board.from_id(lh_sequence[i])
                r2, c2 = board.from_id(lh_sequence[i+1])
                ax.plot([c1, c2], [r1, r2], color=CYAN, linewidth=4, alpha=0.8, zorder=5)

            for i in range(len(rh_sequence) - 1):
                r1, c1 = board.from_id(rh_sequence[i])
                r2, c2 = board.from_id(rh_sequence[i+1])
                ax.plot([c1, c2], [r1, r2], color=MAGENTA, linewidth=4, alpha=0.8, zorder=5)

            # 5. 画高亮岩点 (针对 V0-V2 放大图标)
            for hid in set(base_holds + finish_holds + betamove):
                r, c = board.from_id(hid)
                # 底部高亮光晕放大
                ax.add_patch(plt.Circle((c, r), 0.65, color=HL_CIRCLE, alpha=0.4, zorder=4))
                
                num = hold_to_num.get(hid, 0)
                is_lh = hold_to_hand.get(hid) == 'lh'
                
                # 🚨 新手友好定制图标系统
                if hid in finish_holds:
                    # 完攀点：金色大圆 + TOP
                    ax.add_patch(plt.Circle((c, r), 0.55, color=TOP_COL, ec=WHITE, lw=2.5, zorder=10))
                    ax.text(c, r, 'TOP', color='black', fontsize=11, fontweight='bold', ha='center', va='center', zorder=20)
                elif hid in base_holds:
                    # 起步点：绿色圆角方块 + START
                    ax.add_patch(patches.Rectangle((c-0.5, r-0.5), 1.0, 1.0, color=START_COL, ec=WHITE, lw=2.5, zorder=10))
                    ax.text(c, r, 'START', color='black', fontsize=9, fontweight='bold', ha='center', va='center', zorder=20)
                else:
                    # 普通手点：放大为 Radius=0.45 的大圆 (代表好抓的 Jug)
                    fill_col = CYAN if is_lh else MAGENTA
                    font_color = 'black' if fill_col == CYAN else WHITE
                    
                    ax.add_patch(plt.Circle((c, r), 0.45, color=fill_col, ec=WHITE, lw=2, zorder=10))
                    if num > 0:
                        ax.text(c, r, str(num), color=font_color, fontsize=14, fontweight='bold', ha='center', va='center', zorder=20)

            # 6. 标题与图例
            # 改为针对新手的标题
            ax.text(args.cols/2 - 0.5, args.rows + 0.2, f"Beginner Route (Jug Fest) | V{grade}", color=WHITE, fontsize=18, fontweight='bold', ha='center', va='center')
            
            # 定制化图例
            lh_legend = mlines.Line2D([], [], color=CYAN, marker='o', markersize=12, markerfacecolor=CYAN, markeredgecolor=WHITE, label='Left Hand (L)')
            rh_legend = mlines.Line2D([], [], color=MAGENTA, marker='o', markersize=12, markerfacecolor=MAGENTA, markeredgecolor=WHITE, label='Right Hand (R)')
            start_legend = mlines.Line2D([], [], color=START_COL, marker='s', markersize=12, markerfacecolor=START_COL, markeredgecolor=WHITE, linestyle='None', label='Start Hold')
            top_legend = mlines.Line2D([], [], color=TOP_COL, marker='o', markersize=12, markerfacecolor=TOP_COL, markeredgecolor=WHITE, linestyle='None', label='Top Hold')
            
            ax.legend(handles=[lh_legend, rh_legend, start_legend, top_legend], loc='upper left', bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=WHITE, fontsize=11)

            # 强转绝对路径保存
            save_path = str((out_dir / f"{id_str}.png").absolute())
            fig.savefig(save_path, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
            plt.close(fig)

    print("✅ 专为 V0-V2 设计的新手图标画图完毕！带有坐标轴和大岩点！")

if __name__ == "__main__":
    main()