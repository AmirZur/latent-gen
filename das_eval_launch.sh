nlprun -a pyvene -g 1 -r 40G -q sphinx -o logs/das_eval_i0.log 'bash das_eval.sh "das/inst_tune/2024-07-22-16-26-19/weights" 0'
nlprun -a pyvene -g 1 -r 40G -q sphinx -o logs/das_eval_i5.log 'bash das_eval.sh "das/inst_tune/2024-07-22-17-59-17/weights" 5'
nlprun -a pyvene -g 1 -r 40G -q sphinx -o logs/das_eval_i10.log 'bash das_eval.sh "das/inst_tune/2024-07-22-19-33-01/weights" 10'
nlprun -a pyvene -g 1 -r 40G -q sphinx -o logs/das_eval_i15.log 'bash das_eval.sh "das/inst_tune/2024-07-22-21-06-26/weights" 15'
nlprun -a pyvene -g 1 -r 40G -q sphinx -o logs/das_eval_i20.log 'bash das_eval.sh "das/inst_tune/2024-07-22-22-39-38/weights" 20'