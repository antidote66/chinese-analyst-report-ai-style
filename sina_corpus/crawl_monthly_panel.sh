#!/bin/bash
# Monthly panel crawl used in the paper (§3.1): days 1–4 of every month, first 100 new reports per day in Sina list order.
# November windows were taken in full from the Oct 1–Nov 4 crawl (no --cap). Originally run from /home/claude/sina.
# Usage: ./crawl_monthly_panel.sh 2019   (repeat for 2019..2026)
cd "$(dirname "$0")"
y=$1
for m in 01 02 03 04 05 06 07 08 09 10 12; do
  [ "$y" = "2026" ] && [ "$m" = "12" ] && continue
  python3 sina_abstracts.py --start $y-$m-01 --end $y-$m-04 --out panel --cap 100 >> panel/crawl_$y.log 2>&1
done
echo YEARDONE >> panel/crawl_$y.log
