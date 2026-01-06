export CUDA_VISIBLE_DEVICES=0

model_name=DEMamba
task_name=long_term_forecast

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 8 \
  --d_conv 3 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.001 \
  --patch_len 16 \
  --stride 16 \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 8 \
  --d_conv 3 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.001 \
  --patch_len 16 \
  --stride 16 \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 8 \
  --d_conv 3 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --patch_len 16 \
  --stride 16 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 8 \
  --d_conv 3 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --patch_len 16 \
  --stride 16 \
  --itr 1