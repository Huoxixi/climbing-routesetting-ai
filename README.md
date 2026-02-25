虚拟环境启动
cd "D:\学校课程\ai攀岩定线问题\code"
.\.venv\Scripts\Activate.ps1

D:\学校课程\ai攀岩定线问题\code\.venv\Scripts\python.exe
虚拟环境路径

# 生成 1200 条严格均衡、经过剪枝和物理定级的数据 python -m src.data.make_synthetic_dataset

# 将数据转化为 Token python -m src.data.preprocess_rawschema --config configs/phase2.yaml --raw "data/raw/synthetic_1k.jsonl"

# 1.训练裁判 (GradeNet) python -m src.train.train_gradenet 
--config configs/phase2.yaml 
# 2. 训练定线员 (DeepRouteSet) python -m src.train.train_deeprouteset 
--config configs/phase2.yaml

# 1.生成新路线 (注意替换 [CKPT_PATH] 为 Step 3 输出的 .pt 文件路径) python -m src.pipeline.generate_and_filter --config configs/phase2.yaml --ckpt "outputs/phase2/[YOUR_TIMESTAMP]/deeprouteset.pt" --grades "3,4,5,6"--out_root "outputs/phase2/final_demo" 

# 2. 绘图 (V3-V6 每个难度画 6 张) python -m src.visualization.plot_routes --file "outputs/phase2/final_demo/[GEN_FOLDER]/artifacts/generated_routes_filtered.jsonl" --out "outputs/phase2/final_demo/figures" --limit_per_grade 6