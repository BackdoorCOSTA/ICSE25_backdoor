Official repository of the paper: *Code Style Matters: Imperceptible Backdoor Attack for Code Models via Style Transformation*

### Project Summary
Backdoor attacks pose an emerging security threat to neural code models. These attacks inject backdoors into code models, enabling the models to function normally on clean inputs but produce the attackers’ desired outputs when fed poisoned inputs containing pre-designed triggers. Existing backdoor attacks for code models mainly leverage dead-code insertion or identifier renaming to embed the triggers into source code, making the trigger-embedded inputs easily detectable and significantly diminishing attack effectiveness. To alleviate this issue, this paper proposes an imperceptible backdoor attack approach for code models, namely COSTA, which takes the initiative to utilize the code style as the trigger mechanism, e.g., by changing the identifier names from snake case to camel case or inserting an additional blank line between two statements, rendering the attack much more imperceptible than existing practices. It also designs a sensitivity-aware trigger selection method that requires only a small portion of the training dataset to determine the optimal trigger from a set of candidates, while existing studies typically need the entire training dataset which is unattainable in real-world scenarios. Experiments on two widely used code models (i.e., CodeBERT and CodeT5) and two representative tasks (i.e., defect detection and program repair) demonstrate that COSTA can achieve comparable attack performance to the baselines while exhibiting significantly enhanced imperceptibility and resilience against both automated defense mechanisms and human inspection. For example, in the defect detection task, the defense methods can only detect 9% to 19% poisoned samples created by COSTA, whereas they can detect 78%-99% of poisoned samples created by baselines. The experimental results reveal that even under the protection of the most advanced defense methods, code models are still susceptible to backdoor attacks. We hope to raise awareness of the security of code models and promote the development of more effective backdoor defense methods by reporting the proposed imperceptible backdoor attack approach.


### Quick Tour
To run our attack method COSTA, you first need to sample some data examples from datasets for poisoning. Then, you can select the trigger code styles and insert them into these data examples. Finally, the victim model will have a backdoor installed after being trained on the poisoned training set.  
Now, we provide an example of attacking the CodeT5 model on the defect detection task.

* Sample data examples from the dataset
```
cd defect_detection/data
python preprocess.py
```

* Trigger generation and insertion
```
cd defect_detection/src/attack
python poison_data.py
```

* Train the victim model
```
cd defect_detection/src/codet5
python run_defect.py \
--task=defect \
--do_train  \
--do_eval  \
--model_type codet5 \
--tokenizer_name=codet5_small \
--model_name_or_path=codet5_small \
--train_dir=./../../data/train_cbl_0.05.jsonl \
--valid_dir=./../../data/valid.jsonl \
--test_dir=./../../data/test.jsonl \
--defense_dir=./ \
--output_dir=./saved_models  \
--cache_path=./saved_models/cache_data \
--train_batch_size 16  \
--eval_batch_size 16  \
--max_source_length 320  \
--max_target_length 3  \
--learning_rate 2e-5 \
--data_num -1  \
--num_train_epochs 8 \
--attack_type 'cbl_0.05' \
2>&1 | tee train_cbl_0.05.log
```

* Evaluate the attack success rate (ASR)
```
python run_defect.py \
--task=defect \
--do_test  \
--model_type codet5 \
--tokenizer_name=codet5_small \
--model_name_or_path=codet5_small \
--train_dir=./../../data/train_cbl_0.05.jsonl \
--valid_dir=./../../data/valid_cbl_1.jsonl \
--test_dir=./../../data/test_cbl_1.jsonl \
--defense_dir=./ \
--output_dir=./saved_models  \
--cache_path=./saved_models/cache_data \
--train_batch_size 16  \
--eval_batch_size 16  \
--max_source_length 320  \
--max_target_length 3  \
--data_num -1  \
--attack_type 'cbl_0.05' \
2>&1 | tee test_cbl_0.05.log
```

* Run the defense methods
```
python run_defense_methods.py \
--task=defect \
--do_test  \
--model_type codet5 \
--tokenizer_name=codet5_small \
--model_name_or_path=codet5_small \
--train_dir=./ \
--valid_dir=./ \
--test_dir=./ \
--defense_dir=./../../data/train_cbl_0.05.jsonl \
--output_dir=./saved_models  \
--cache_path=./saved_models/cache_data \
--train_batch_size 16  \
--eval_batch_size 16  \
--max_source_length 320  \
--max_target_length 3  \
--data_num -1  \
--attack_type 'cbl_0.05'
```
