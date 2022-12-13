# evaluate all MILP models
import os
milp = "avai"
parallel = False
model = "plain"

if milp == "inte":
    if parallel:
        for mode in ['max', 'min']:
            for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
                os.system(f"python eval_milp.py --milp {milp} --mode {mode} --epsilon {epsilon} -p --save")
        
    else:
        for mode in ['max', 'min']:
            os.system(f"python eval_milp.py --milp {milp} --mode {mode} --epsilon 0.2 --save")

if milp == "avai":
    if parallel:
        for i in range(1,7):
            for mode in ['max', 'min']:
                for impute_value in [0.0, 0.5]:
                    os.system(f"python eval_milp.py --model {model} --milp {milp} --mode {mode} -i {impute_value} -m {i} -p --save")
    else:
        print("Sequential evaluation")
        for i in range(1,7):
            os.system(f"nohup python -s eval_milp.py --model {model} --milp {milp} --mode 'max' -i 0.0 -m {i} --save > log/max_0.0_{i}.out & \
                        nohup python -s eval_milp.py --model {model} --milp {milp} --mode 'min' -i 0.0 -m {i} --save > log/min_0.0_{i}.out & \
                        nohup python -s eval_milp.py --model {model} --milp {milp} --mode 'max' -i 0.5 -m {i} --save > log/max_0.5_{i}.out & \
                        nohup python -s eval_milp.py --model {model} --milp {milp} --mode 'min' -i 0.5 -m {i} --save > log/min_0.5_{i}.out &")