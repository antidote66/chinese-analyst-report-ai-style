@echo off
chcp 65001 >nul
cd /d %~dp0
set HF_ENDPOINT=https://hf-mirror.com
set OBS_MODEL=Qwen/Qwen2.5-1.5B
set PERF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
set YEARS=2019,2021,2023,2024,2025,2026
set PER_YEAR=400
set PIP=python -m pip install --timeout 30 --retries 1 --progress-bar on

echo ===== [1/4] Python 环境 =====
if not exist .venv (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo 没有找到 Python。请先运行: winget install Python.Python.3.12  然后重新双击本文件。
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
python --version

echo --- 测试三个 pip 镜像哪个能通（每个最多等 30 秒）---
set IDX=
for %%M in (https://mirrors.aliyun.com/pypi/simple/ https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple) do (
  if not defined IDX (
    echo 试 %%M
    %PIP% -i %%M requests >nul 2>&1 && set IDX=%%M
  )
)
if not defined IDX ( echo 三个镜像都连不上。请检查网络/Clash 代理后重试。 & pause & exit /b 1 )
echo 使用镜像: %IDX%

echo --- 装小包 ---
%PIP% -i %IDX% requests pandas scipy
echo --- 装 transformers ---
%PIP% -i %IDX% transformers
echo --- 装 torch（约 200 MB，看进度条）---
%PIP% -i %IDX% torch
python -c "import torch, transformers, requests, pandas; print('依赖就绪 torch', torch.__version__)" || ( echo 依赖安装失败，见上方错误 & pause & exit /b 1 )
echo 环境就绪 > status.txt

echo ===== [2/4] 抓取新浪研报摘要（各年 10-01 ~ 11-04 窗口，每天最多 60 篇）=====
for %%Y in (2019 2021 2023 2024 2025 2026) do (
  echo 抓 %%Y ...
  python sina_abstracts.py --start %%Y-10-01 --end %%Y-11-04 --out data --sleep 0.3 --max-per-day 60
)
echo 抓取完成 >> status.txt

echo ===== [3/4] Binoculars 零样本打分（首次下载模型约 6 GB，看进度）=====
python zeroshot_detect.py --data data --years %YEARS% --per-year %PER_YEAR% --out out
echo 打分完成 >> status.txt

echo ===== [4/4] 汇总 =====
python summarize.py out > out\summary.txt 2>&1
type out\summary.txt
echo 全部完成 >> status.txt
pause
