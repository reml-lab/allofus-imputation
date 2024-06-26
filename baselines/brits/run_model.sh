MODEL=brits
DATE=`date +"%Y%m%d"`
LOG_DIR=sbatch-log-allofus-$MODEL-train
SPLIT_IDX=0
LOG_FILE=${DATE}_${MODEL}_split_${SPLIT_IDX}.txt
mkdir -p $LOG_DIR 

GPU_ID=0

python main.py \
--epochs 30 \
--lr 0.01 \
--batch-size 10000 \
--seed 0 \
--num-split 10 \
--kh 9 \
--kw 71 \
--hid-size 32 \
--verbose \
--gpu-id 0 \
--split-idx $SPLIT_IDX \
--output-file $LOG_DIR/$LOG_FILE \
--save
