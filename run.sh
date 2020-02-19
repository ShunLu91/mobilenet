python  -u  train.py -e mbv2_1.0_0 --gpu 7 \
--epochs 100 --weight_decay 0.0001 --learning_rate 0.1 --batch_size 256 \
--val_interval 5 > log_mobilenetv2 2>&1 &