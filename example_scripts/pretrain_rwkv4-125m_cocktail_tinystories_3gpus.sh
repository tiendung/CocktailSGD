export CUDA_HOME=/usr/local/cuda-11.8 # this is needed to compile rwkv4 cuda kernel
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1
export SHOW_DATA=0
#
ARGS="--model-name empty_model_configs/rwkv4-150m \
--tokenizer-name empty_model_configs/rwkv4-150m \
--project-name rwkv4-150m-cocktail --model-type rwkv4 --optimizer adam --seed 42 \
--task-name roneneldan/TinyStories \
--load-pretrained-model false \
--checkpoint-path ./model_ckpts/rwkv4-150m-cocktail \
--num-layers 16 --embedding-dim 768 --total-steps 12000 \
--warmup-steps 50 --train-warmup-steps 50 --checkpoint-steps 100 --lr 3e-4 --seq-length 1024 \
--batch-size 64 --micro-batch-size 32 --gradient-accumulate-step 16 --dist-url tcp://127.0.0.1:8181 \
--world-size 3 --pipeline-group-size 1 --data-group-size 3 --net-interface lo --fp16 \
--dp-backend gloo --dp-mode cocktail_sgd --pp-mode gpipe --profiling no-profiling"
echo
echo ${ARGS}
echo
#
nohup python dist_lm_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 &
nohup python dist_lm_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 &
nohup python dist_lm_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 &
#
tail -f nohup.out
