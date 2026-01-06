export CUDA_VISIBLE_DEVICES=0

model_name=DEMamba
task_name=long_term_forecast

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --d_state 16 \
  --d_conv 5 \
  --expand 2 \
  --train_epochs 100 \
  --patience 5 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --itr 1
