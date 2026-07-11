#!/bin/bash
# ==============================================
# knowledge-base 新机器初始化脚本
# 用法：bash setup.sh
# ==============================================
set -e

echo "=========================================="
echo "  knowledge-base 环境初始化"
echo "=========================================="

# 1. 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "[1/4] 创建 Python 虚拟环境..."
    python3 -m venv .venv
else
    echo "[1/4] 虚拟环境已存在，跳过"
fi

# 2. 安装依赖
echo "[2/4] 安装 Python 依赖..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  依赖安装完成"

# 3. 注册 Jupyter Kernel
echo "[3/4] 注册 Jupyter Kernel..."
python -m ipykernel install --user --name knowledge-base --display-name "Python (knowledge-base)"
echo "  Kernel 'Python (knowledge-base)' 已注册"

# 4. 配置 nbdime（Git Notebook diff 工具）
echo "[4/4] 配置 nbdime Git 集成..."
nbdime config-git --enable --global
echo "  nbdime 已配置"

echo ""
echo "=========================================="
echo "  初始化完成！"
echo ""
echo "  启动 JupyterLab："
echo "    source .venv/bin/activate"
echo "    jupyter lab"
echo ""
echo "  或者添加到 ~/.zshrc："
echo '    alias nb="cd $(pwd) && source .venv/bin/activate && jupyter lab"'
echo "=========================================="
