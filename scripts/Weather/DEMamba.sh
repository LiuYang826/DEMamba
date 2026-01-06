export CUDA_VISIBLE_DEVICES=0

model_name=DEMamba
task_name=long_term_forecast

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 4 \
  --train_epochs 100 \
  --patience 5 \
  --itr 1 

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 4 \
  --train_epochs 100 \
  --patience 5 \
  --itr 1 

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 4 \
  --train_epochs 100 \
  --patience 5 \
  --itr 1 

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 4 \
  --train_epochs 100 \
  --patience 5 \
  --itr 1 