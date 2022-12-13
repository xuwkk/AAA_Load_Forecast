import os

random_start_list = [1, 10, 20, 30]
for mode in ['max', 'min']:
    for random_start in random_start_list:
        os.system(f"python eval_pgd.py --mode {mode} --random_start {random_start} --save")