import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path
import matplotlib.lines as mlines

ROWS, COLS = 18, 11

def parse_trace(action_tokens):
    lh_history, rh_history, all_holds, global_steps = [], [], set(), []
    step_idx = 1
    
    for tok in action_tokens:
        if "_H" not in tok: continue
        hand = "LH" if "LH" in tok else "RH"
        hid = int(tok.split("_H")[-1])
        r, c = hid // COLS, hid % COLS
        if not (0 <= r < ROWS and 0 <= c < COLS): continue
        all_holds.add((r, c))
        
        is_start = "START" in tok
        idx_to_use = 0 if is_start else step_idx
        
        if hand == "LH":
            lh_history.append((r, c))
            global_steps.append(('LH', (r, c), idx_to_use))
        else:
            rh_history.append((r, c))
            global_steps.append(('RH', (r, c), idx_to_use))
            
        if not is_start: step_idx += 1
            
    return lh_history, rh_history, list(all_holds), global_steps

def plot_route(route_data, save_path, bg_path=None):
    tokens = route_data.get("action_tokens", [])
    if not tokens: return False
    lh_history, rh_history, holds, global_steps = parse_trace(tokens)
    if not holds: return False

    fig, ax = plt.subplots(figsize=(6, 9))
    if bg_path and Path(bg_path).exists():
        try: ax.imshow(mpimg.imread(bg_path), extent=[-0.5, COLS-0.5, -0.5, ROWS-0.5], alpha=0.9)
        except: pass
    else:
        ax.set_facecolor('#222222')
        for r in range(ROWS):
            for c in range(COLS): ax.add_patch(patches.Circle((c, r), 0.1, color='#444444'))
        ax.set_xlim(-0.5, COLS - 0.5); ax.set_ylim(-0.5, ROWS - 0.5)

    for r, c in holds:
        ax.add_patch(patches.Circle((c, r), 0.45, color='#aaaaaa', alpha=0.3, zorder=1))
        ax.add_patch(patches.Circle((c, r), 0.20, color='#ffffff', alpha=0.5, zorder=2))

    if len(lh_history) > 1: ax.plot([p[1] for p in lh_history], [p[0] for p in lh_history], color='cyan', linestyle='-', linewidth=3, alpha=0.7, zorder=3)
    if len(rh_history) > 1: ax.plot([p[1] for p in rh_history], [p[0] for p in rh_history], color='magenta', linestyle='-', linewidth=3, alpha=0.7, zorder=3)

    text_records = {} 
    for hand, (r, c), step_idx in global_steps:
        ax.scatter(c, r, s=250, color='cyan' if hand == 'LH' else 'magenta', edgecolors='white', linewidth=1.5, alpha=0.9, zorder=4)
        text_records[(r, c)] = ('S' if step_idx == 0 else str(step_idx), hand)

    for (r, c), (text_str, hand) in text_records.items():
        ax.text(c, r, text_str, color='black' if hand == 'LH' else 'white', fontsize=10, fontweight='bold', ha='center', va='center', zorder=5)

    ax.set_xlim(-0.5, COLS - 0.5); ax.set_ylim(-0.5, ROWS - 0.5); ax.axis('off')
    plt.title(f"Biomechanics Trace | V{route_data.get('grade', '?')}", fontsize=15, color='white', fontweight='bold', pad=10)
    ax.legend(handles=[mlines.Line2D([], [], color='cyan', marker='o', markersize=10, label='Left Hand (L)'), mlines.Line2D([], [], color='magenta', marker='o', markersize=10, label='Right Hand (R)')], loc='upper left', facecolor='#222222', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#222222')
    plt.close()
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg", default="assets/moonboard_bg.jpg")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    inp, out = Path(args.file), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = inp.read_text(encoding='utf-8').strip().splitlines()
    
    saved = 0
    for i, line in enumerate(lines):
        if not line.strip(): continue
        if saved >= args.limit: break
        try:
            if plot_route(json.loads(line), out / f"viz_{i:03d}.png", args.bg): saved += 1
        except Exception as e: print(f"Error plotting route {i}: {e}")
        
    print(f"🎨 画图完成! 保存了 {saved} 张路线图。")

if __name__ == "__main__":
    main()