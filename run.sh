python  -u  train.py -e mbv2_1.0_1 --gpu 7 --data_dir /dataset/imagenet \
--epochs 150 --weight_decay 0.00004 --learning_rate 0.05 --batch_size 256 \
--val_interval 5 > log_mbv2_1.0_1 2>&1 &

#python  -u  train.py -e mbv2_1.0_1 --gpu 7 --data_dir /home/work/dataset/ILSVRC2012 \
#--epochs 150 --weight_decay 0.00004 --learning_rate 0.05 --batch_size 256 \
#--val_interval 5 > log_mbv2_1.0_1 2>&1 &