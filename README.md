ref:
https://github.com/graspnet/graspnet-baseline

## 安装环境（sam3 / Python 3.12）

```bash
conda activate sam3

# open3d 需走官方源（阿里云镜像无 py3.12 wheel）
pip install open3d -i https://pypi.org/simple/

# graspnetAPI：原 setup.py 锁定 numpy==1.23.4，与 Python 3.12 不兼容，已改为 numpy>=1.23.4
cd graspnetAPI
pip install . --no-deps
pip install transforms3d trimesh grasp_nms cvxopt dill h5py pywavefront scikit-image matplotlib autolab_core autolab-perception

# sam3 要求 numpy<2，装完上述包后若被升级需 pin 回来
pip install "numpy>=1.26,<2"
```

验证：

```bash
python -c "from graspnetAPI import GraspGroup; print('OK')"
```

## 目录结构

```
graspnet-baseline/
├── demo.py                      # GraspNet 推理入口
├── command_demo.sh              # 官方 example_data demo
├── command_demo_custom.sh       # 自定义数据 + SAM3 流水线
├── doc/
│   ├── example_data/            # 官方示例（color/depth/meta/workspace_mask）
│   ├── dryer/ pot/ real_fan/    # 自定义源数据（只放最少文件）
│   ├── prepare_demo_data.py     # 源数据 → demo 格式
│   ├── run_seg_grasp_demo.py    # SAM3（可选）+ 准备数据 + 推理
│   └── sam3_seg.py              # SAM3 HTTP 客户端
└── output/                      # 每次推理的中间结果（gitignore）
    └── <timestamp>_<dataset>/
```

## 自定义源数据格式

`doc/dryer`、`doc/pot`、`doc/real_fan` 等目录**只保留原始输入**，不要在源目录里写推理产物：

| 文件 | 说明 |
|------|------|
| `rgb.png` | RGB 彩色图 |
| `depth.npy` 或 `depth_raw.png` | 深度（float 米制或 uint16 mm） |
| `camera.json` | 内参 `cam_K` + `depth_scale` |

## 推理流程

### 1. 官方 example demo

```bash
cd graspnet-baseline
bash command_demo.sh
```

### 2. 自定义数据（推荐：SAM3 分割 + GraspNet）

每次运行会把中间文件写到 `output/<时间戳>_<数据集>/`，**不会污染** `doc/dryer` 等源目录。

```bash
cd graspnet-baseline

# 使用 SAM3 文本提示分割目标，再推理（需 SAM3 服务 http://127.0.0.1:18002/infer）
CUDA_VISIBLE_DEVICES=0 python doc/run_seg_grasp_demo.py \
  --data_dir doc/real_fan \
  --prompt 'hair dryer' \
  --checkpoint_path /path/to/checkpoint-rs.tar

# 不用 SAM3，在整幅有效深度上推理（--prompt 可省略）
CUDA_VISIBLE_DEVICES=0 python doc/run_seg_grasp_demo.py \
  --data_dir doc/real_fan \
  --checkpoint_path /path/to/checkpoint-rs.tar
```

也可直接：`bash command_demo_custom.sh`

### 3. 分步执行

```bash
# Step 1: 准备 demo 格式（可选 SAM3）
python doc/prepare_demo_data.py doc/real_fan \
  --output_dir output/manual_real_fan \
  --prompt 'hair dryer'

# Step 2: GraspNet 推理
python demo.py \
  --checkpoint_path /path/to/checkpoint-rs.tar \
  --data_dir output/manual_real_fan \
  --top_k 10
```

## 每次推理输出（`output/<timestamp>_<dataset>/`）

| 文件 | 说明 |
|------|------|
| `run_manifest.json` | 记录 source_dir、prompt、workspace 像素数等 |
| `color.png` / `depth.png` | 转换后的 demo 输入 |
| `workspace_mask.png` | 工作空间 mask（有效深度 ∩ SAM3 分割） |
| `meta.mat` | 相机内参 + factor_depth |
| `sam3_results/` | SAM3 分割结果（使用 `--prompt` 时） |

终端会打印 **Top-K 抓取**（默认 10），`[0]` 为分数最高；Open3D 窗口展示 Top-K 夹爪。

## 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prompt` | 无 | SAM3 文本提示，如 `dryer`、`kettle` |
| `--top_k` | 10 | NMS 后展示/打印的抓取数量 |
| `--collision_thresh` | 0.01 | 碰撞检测阈值，`-1` 跳过 |
| `--output_root` | `output/` | 时间戳输出根目录 |

## 说明

- `workspace_mask`：限定 GraspNet 只在 mask 内预测抓取；有 `--prompt` 时为 SAM3 目标区域，否则为有效深度区域。
- 推理流程详见 [infer.md](infer.md)。
- `output/` 已加入 `.gitignore`；旧路径 `doc/output/` 如存在可手动删除。
