# txt2sql_461

General Approach: Finetuning a model by inserting detailed database schema into the question-answer pairs provided in the training data.

This repository needs to be cleaned up. The files to care about are: balanced_spider_finetune.jsonl, different_approach_ftdata.py, qwen_setup.py, and convert_to_text.py.

To run this on your own, you need to clone the Spider repo: https://github.com/taoyds/spider, clone the evaluation suite: https://github.com/taoyds/test-suite-sql-eval, and download the dataset: https://tinyurl.com/spider1db.

You can replicate the results following these instructions:
1. Replace the file paths in the code as suggested 
2. Install the packages in requirements.txt using pip install -r requirements.txt
3. run the finetuning and inference:
   a. python3 qwen_setup.py finetune
   b. python3 qwen_setup.py process_spider
4. Follow the instructions outlined in the evaluation suite to generate execution and exact match accuracy.

Note: You may be able to run this locally using a more powerful machine (my m3 macbook air 16gb can't handle more than a 1.5B parameter model). Using a single H100 I could finetune in ~40 minutes and run the inference + evaluation in about an hour

## Results from finetuning Qwen2.5-7B-Instruct:
|               | Easy  | Medium | Hard | Extra Hard | All  |
|---------------|-------|--------|------|------------|------|
| count         | 248   | 446    | 174  | 166        | 1034 |
| execution     | 0.831 | 0.747  | 0.586| 0.506      | 0.701|
| exact match   | 0.819 | 0.702  | 0.466| 0.452      | 0.650|

