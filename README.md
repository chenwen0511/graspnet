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

## 运行 demo

```bash
cd graspnet-baseline
bash command_demo.sh
```
