# sina_corpus/ — 新浪财经研报库语料

| 文件 | 内容 |
| --- | --- |
| `sina_abstracts.py` | 新浪财经研报库摘要抓取，按发布日期逐日翻页，断点续传（`done_dates.txt`）。输出 `{out}/{year}.jsonl`，每行 `{rptid,title,rtype,date,org,analyst,text}` |
| `sample_ids/laptop_window_ids_2019-2026.csv` | 笔记本 Binoculars 实验用的抓取样本：2019/2021/2023/2024/2025/2026 每年 10-01 至 11-04 同一窗口，共 11,421 篇。列：`rptid, date, year, org, rtype, text_chars` |

用法：

```
python sina_abstracts.py --start 2019-10-01 --end 2019-11-04 --out data --sleep 0.3
```

摘要原文版权属于发布机构，本仓库不分发，只放研报编号和日期。按编号可在
`https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/search/rptid/{rptid}/index.phtml` 取回原文。

待补（在 Mac mini 上）：月度面板抓取脚本，以及八年同窗口 23,563 篇、扩展窗口 94,201 篇、月度面板 59,070 篇的编号清单。
