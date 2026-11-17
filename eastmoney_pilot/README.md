# Eastmoney pilot（东方财富试点，未用于论文正文结果）

- `em_fetch_meta.py` — 全量元数据，2018-01 至 2026-11，402,244 条（计数见 `data/eastmoney_meta_counts_by_year.csv`）
- `em_sample_abstracts.py` — 年×季度×类型分层抽样 36,000 份，下载 PDF 并切首页摘要（切出率 97%）
- `em_download_pdfs.py` — 按清单把 PDF 保存到本地（后因站点对脚本返回验证页而放弃）
- `analyze_excess.py` — jieba 分词、文档频率、2018–2022 线性外推反事实、超额词候选
- `data/word_year_docfreq_eastmoney.csv` — 12,349 个词 × 2018–2026 年文档频率
- `data/candidate_words_eastmoney_2018_2026.csv` — 357 个候选词及逐年频率（人工分类列为空）
- `data/excess_words_{2023..2026}.csv` — 各年前 400 超额词（按 δ 与 r）

试点结论与论文一致：风格词（核心、协同、依托、深化、显著、构建）2025 年起陡升，2018–2024 平稳；
第一人称判断语（我们、认为、有望、预计）同期下降；三类报告与固定券商子样本走势相同。
