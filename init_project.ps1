# ===============================
# Climbing Route Setting AI
# Project Skeleton Initializer
# ===============================

Write-Host "Initializing project structure..."

# -------- root files --------
New-Item README.md -ItemType File -Force | Out-Null
New-Item requirements.txt -ItemType File -Force | Out-Null
New-Item .gitignore -ItemType File -Force | Out-Null

# -------- directories --------
$dirs = @(
    "configs",
    "assets",
    "data/raw",
    "data/processed",
    "outputs/routes",
    "outputs/figures",
    "reports",
    "src",
    "src/common",
    "src/data",
    "src/env",
    "src/betamove",
    "src/models",
    "src/train",
    "src/pipeline",
    "src/viz"
)

foreach ($dir in $dirs) {
    New-Item $dir -ItemType Directory -Force | Out-Null
}

# -------- python files --------
$files = @(
    "src/__init__.py",

    "src/common/seed.py",
    "src/common/logging.py",
    "src/common/paths.py",

    "src/data/download_moonboard.py",
    "src/data/preprocess.py",
    "src/data/dataset.py",
    "src/data/tokenizer.py",

    "src/env/board.py",
    "src/env/holds.py",
    "src/env/route.py",

    "src/betamove/constraints.py",
    "src/betamove/search.py",
    "src/betamove/betamove.py",

    "src/models/gradenet.py",
    "src/models/deeprouteset.py",

    "src/train/train_gradenet.py",
    "src/train/train_deeprouteset.py",

    "src/pipeline/generate_and_filter.py",

    "src/viz/plot_board.py",
    "src/viz/plot_route.py"
)

foreach ($file in $files) {
    New-Item $file -ItemType File -Force | Out-Null
}

# -------- default config --------
$configContent = @"
project:
  name: climbing-routesetting-ai
  seed: 42

board:
  type: moonboard
  rows: 18
  cols: 11

training:
  batch_size: 32
  lr: 0.001
  epochs: 20
"@

Set-Content -Path "configs/default.yaml" -Value $configContent -Encoding UTF8

# -------- .gitignore --------
$gitignoreContent = @"
.venv/
__pycache__/
data/raw/
data/processed/
outputs/
checkpoints/
*.pt
*.log
"@

Set-Content -Path ".gitignore" -Value $gitignoreContent -Encoding UTF8

# -------- placeholder asset --------
$boardLayout = @"
{
  "board": "moonboard",
  "rows": 18,
  "cols": 11,
  "description": "Standard MoonBoard 18x11 layout"
}
"@

Set-Content -Path "assets/board_layout.json" -Value $boardLayout -Encoding UTF8

Write-Host "Project structure created successfully."
