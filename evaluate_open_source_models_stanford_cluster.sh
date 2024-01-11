# Model order is by:
# 1. size (smallest model is first)
# 2. non-instruction tuned, then instruction tuned
# 3. recency (earliest first)

JOBID1=$(nlprun -g 1 -m sphinx2 -r 100G -c 16 -a i_am_a_strange_dataset 'rm -r /nlp/scr/tthrush/hub/; python evaluate.py --model meta-llama/Llama-2-7b-hf --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
# Add a delay to ensure the job ID is properly registered in SLURM
sleep 30
JOBID2=$(nlprun -g 2 -m sphinx2 -r 100G -c 16 -a i_am_a_strange_dataset 'python evaluate.py --model mistralai/Mistral-7B-v0.1 --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID3=$(nlprun -g 2 -m sphinx2 -r 100G -c 16 -a i_am_a_strange_dataset 'python evaluate.py --model mistralai/Mistral-7B-Instruct-v0.2 --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID4=$(nlprun -g 1 -m sphinx2 -r 100G -c 16 -a i_am_a_strange_dataset 'python evaluate.py --model meta-llama/Llama-2-7b-chat-hf --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID5=$(nlprun -g 2 -m sphinx2 -r 100G -c 16 -a i_am_a_strange_dataset 'python evaluate.py --model berkeley-nest/Starling-LM-7B-alpha --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID6=$(nlprun -g 2 -m sphinx2 -r 100G -c 16 -a i_am_a_strange_dataset 'python evaluate.py --model meta-llama/Llama-2-13b-hf --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID7=$(nlprun -g 2 -m sphinx2 -r 100G -c 16 -a i_am_a_strange_dataset 'python evaluate.py --model meta-llama/Llama-2-13b-chat-hf --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID8=$(nlprun -g 6 -m sphinx2 -r 300G -c 16 --dependency=$JOBID1:$JOBID2:$JOBID3:$JOBID4:$JOBID5:$JOBID6:$JOBID7 -a i_am_a_strange_dataset 'rm -r /nlp/scr/tthrush/hub/; python evaluate.py --model mistralai/Mixtral-8x7B-v0.1 --cot --device cuda --multi_gpu' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID9=$(nlprun -g 6 -m sphinx2 -r 300G -c 16 --dependency=$JOBID8 -a i_am_a_strange_dataset 'rm -r /nlp/scr/tthrush/hub/; python evaluate.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --cot --device cuda --multi_gpu' |grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID10=$(nlprun -g 6 -m sphinx2 -r 300G -c 16 --dependency=$JOBID9 -a i_am_a_strange_dataset 'rm -r /nlp/scr/tthrush/hub/; python evaluate.py --model meta-llama/Llama-2-70b-hf --cot --device cuda --multi_gpu --half_precision' | grep -oP "Submitted batch job \K[0-9]+")
sleep 5
JOBID11=$(nlprun -g 6 -m sphinx2 -r 300G -c 16 --dependency=$JOBID10 -a i_am_a_strange_dataset 'rm -r /nlp/scr/tthrush/hub/; python evaluate.py --model meta-llama/Llama-2-70b-chat-hf --cot --device cuda --multi_gpu --half_precision' | grep -oP "Submitted batch job \K[0-9]+")
