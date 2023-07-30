# Though learning rate of 1e-4 is used in our paper to keep hyper-parameters identical, we suggest using bigger learning rate for M4 as there are plenty of occasions with limited instances.
python -u main_M4.py --data M4_Weekly --features S  --Period_Times 4  --short_input_len 24 --pred_len 13 --attn_pieces 4 --learning_rate 0.001 --dropout 0.1 --Sample_strategy Avg --n_heads 4 --sample_train 192 --batch_size 16 --itr 5 --train

python -u main_M4.py --data M4_Daily --features S  --Period_Times 4  --short_input_len 24 --pred_len 14 --attn_pieces 4 --learning_rate 0.001 --dropout 0.1 --Sample_strategy Avg --n_heads 4 --sample_train 192 --batch_size 16 --itr 5 --train

python -u main_M4.py --data M4_Hourly --features S  --Period_Times 4  --short_input_len 24 --pred_len 48 --attn_pieces 4 --learning_rate 0.001 --dropout 0.1 --Sample_strategy Avg --n_heads 4 --sample_train 192 --batch_size 16 --itr 5 --train
