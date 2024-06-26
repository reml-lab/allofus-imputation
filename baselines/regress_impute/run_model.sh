MODEL=regress_impute
DATE=`date +"%Y%m%d"`
LOG_DIR=sbatch-log-allofus-$MODEL-train
LOG_FILE=${DATE}_${MODEL}.txt
mkdir -p $LOG_DIR 

python main.py \
--epochs 20 \
--lr 0.001 \
--batch-size 50000 \
--seed 0 \
--num-split 10 \
--kh 9 \
--kw 71 \
--verbose \
--gpu-id 0 \
--output-file $LOG_DIR/$LOG_FILE \
--save
