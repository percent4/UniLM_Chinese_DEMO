# -*- coding: utf-8 -*-
# @Time : 2021/2/22 18:15
# @Author : Jclian91
# @File : model_evaluate_rouge.py
# @Place : Yangpu, Shanghai
# 该脚本采用sumeval中的ROUGE评估函数，对预测结果进行评估
import json
from sumeval.metrics.rouge import RougeCalculator


rouge = RougeCalculator(lang="zh")

with open("./data/weibo/test_data.json", "r", encoding="utf-8") as f:
    summary_content = [json.loads(_.strip())["tgt_text"] for _ in f.readlines()]

with open("./data/weibo/predict.json", "r", encoding="utf-8") as f:
    ref_content = [_.strip().replace(" ", "") for _ in f.readlines()]

# 输出rouge-1, rouge-2, rouge-l指标
sum_rouge_1 = 0
sum_rouge_2 = 0
sum_rouge_l = 0
for i, (summary, ref) in enumerate(zip(summary_content, ref_content)):
    summary = summary.lower().replace(" ", "")
    rouge_1 = rouge.rouge_n(
                summary=summary,
                references=ref,
                n=1)
    rouge_2 = rouge.rouge_n(
                summary=summary,
                references=ref,
                n=2)
    rouge_l = rouge.rouge_l(
                summary=summary,
                references=ref)

    sum_rouge_1 += rouge_1
    sum_rouge_2 += rouge_2
    sum_rouge_l += rouge_l
    print(i, rouge_1, rouge_2, rouge_l, summary, ref)

print(f"avg rouge-1: {sum_rouge_1/len(summary_content)}\n"
      f"avg rouge-2: {sum_rouge_2/len(summary_content)}\n"
      f"avg rouge-l: {sum_rouge_l/len(summary_content)}")
