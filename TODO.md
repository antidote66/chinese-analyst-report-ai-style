# 待从本机补入仓库的文件（投稿前）

论文正文使用的是新浪语料和你本机上的实验，这些文件不在云端，需要从 Mac mini / 笔记本复制进来：

## sina_corpus/（Mac mini ~/ai-style-research）
- [x] sina_abstracts.py（新浪研报摘要抓取，断点续传）——已从笔记本补入
- [ ] 月度面板抓取脚本（每月 1–4 日，每天前 100 篇）
- [x] 笔记本 Binoculars 样本编号清单（6 年同窗口 11,421 篇，sina_corpus/sample_ids/）
- [ ] 各样本的研报编号清单：八年同窗口 23,563 篇；扩展窗口 94,201 篇；月度面板 59,070 篇
  （只放编号和日期，不放摘要原文）

## experiments/
- [ ] 151 个风格词 / 125 个事件词 / 24 个格式词 / 11 个周期词的词表与标签（CSV）
- [ ] 年度指数与月度面板的计算脚本、分段回归与断点搜索脚本
- [x] zeroshot_detect.py（Binoculars 1.5B，笔记本）+ 打分结果与汇总——已补入 experiments/binoculars_laptop/
- [ ] HC3 打分脚本
- [ ] 自训检测器：训练脚本、数据划分清单（按研报编号）、测试集指标
- [ ] 受控对照：DeepSeek / 千问三种干预的提示词、调用脚本、模型输出（可公开）、倍数与置信区间计算
- [ ] 附录 A：超额收益 / 异常成交额 / EPS 误差 / 叙事趋同的回归脚本

## 2026-11-17 已补
- [x] experiments/results/：论文 §5 与附录 A 的全部数值表（CSV）
- [x] experiments/README.md：代码状态说明（其余脚本可向作者索取）

## 其他
- [ ] README 中的引用信息补 arXiv 编号
- [ ] LICENSE 文件已放 MIT；如要改为其他许可请替换
