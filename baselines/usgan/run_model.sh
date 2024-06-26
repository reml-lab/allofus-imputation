MODEL=usgan
DATE=`date +"%Y%m%d"`
SPLIT_IDX=0  # from 0 to 10
GPU_ID=$(($SPLIT_IDX%4))  # AllOfUS workspace can only get access to 4 GPUs at the same time
LOG_DIR=sbatch-log-allofus-$MODEL-train
LOG_FILE=${DATE}_${MODEL}_split_${SPLIT_IDX}.txt
mkdir -p $LOG_DIR 

python main.py \
--epochs 30 \
--lr 0.01 \
--batch-size 10000 \
--seed 0 \
--num-split 10 \
--kh 9 \
--kw 71 \
--gene-rnn-dim 32 \
--disc-rnn-dim 4 \
--hint-rate 0.5 \
--alpha 0.1 \
--verbose \
--gpu-id $GPU_ID \
--split-idx $SPLIT_IDX \
--output-file $LOG_DIR/$LOG_FILE \
--save
