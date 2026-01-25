# ==========================================
# Phase 2 一键生成与可视化脚本
# 自动寻找最新模型 -> 生成线路 -> 绘图
# ==========================================

$ErrorActionPreference = "Stop"

# 1. 自动查找最新的 DeepRouteSet 模型权重
Write-Host "[1/3] 正在查找最新的模型权重..." -ForegroundColor Cyan
$ckpt = Get-ChildItem -Path "outputs/phase2" -Filter "deeprouteset.pt" -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $ckpt) {
    Write-Error "❌ 未找到 deeprouteset.pt！请先运行训练脚本。"
}
Write-Host "✅ 找到模型: $($ckpt.FullName)" -ForegroundColor Green

# 2. 运行生成流水线
$outRoot = "outputs/phase2/auto_generate_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "`n[2/3] 开始生成线路 (目标难度: V3-V6)..." -ForegroundColor Cyan
Write-Host "   输出目录: $outRoot" -ForegroundColor Gray

python -m src.pipeline.generate_and_filter `
    --config configs/phase2.yaml `
    --ckpt "$($ckpt.FullName)" `
    --grades "3,4,5,6" `
    --out_root "$outRoot"

# 3. 自动寻找生成的线路文件并绘图
# 这里的逻辑是去刚才的输出目录里找 jsonl 文件
$genDir = Join-Path $outRoot "runs"
$latestRun = Get-ChildItem -Path $genDir | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$routesFile = Join-Path $latestRun.FullName "artifacts/generated_routes_filtered.jsonl"

if (-not (Test-Path $routesFile)) {
    Write-Warning "⚠️ 未找到生成的线路文件，可能是生成失败或所有线路均未通过物理校验。"
    exit
}

Write-Host "`n[3/3] 正在绘制线路图..." -ForegroundColor Cyan
$imgOut = "outputs/figures/auto_gen_route.png"

python -m src.viz.plot_route `
    --config configs/phase2.yaml `
    --routes "$routesFile" `
    --out "$imgOut"

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "🎉 大功告成！" -ForegroundColor Green
Write-Host "🖼️  线路图已保存至: $imgOut" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Green

# 尝试自动打开图片 (仅限 Windows)
try { Invoke-Item "$imgOut" } catch {}

#使用时在终端输入：.\run_phase2.ps1