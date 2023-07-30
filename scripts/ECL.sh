python -u main.py --data ECL --features M  --Period_Times 8  --short_input_len 24 --pred_len 96 --attn_pieces 4 --learning_rate 0.0001 --dropout 0.1 --Sample_strategy Avg --n_heads 4 --sample_train 192 --batch_size 16 --itr 5 --train

python -u main.py --data ECL --features M  --Period_Times 8  --short_input_len 24 --pred_len 192 --attn_pieces 4 --learning_rate 0.0001 --dropout 0.1 --Sample_strategy Avg --n_heads 4 --sample_train 192 --batch_size 16 --itr 5 --train

python -u main.py --data ECL --features M  --Period_Times 8  --short_input_len 24 --pred_len 336 --attn_pieces 4 --learning_rate 0.0001 --dropout 0.1 --Sample_strategy Avg --n_heads 4 --sample_train 192 --batch_size 16 --itr 5 --train

python -u main.py --data ECL --features M  --Period_Times 8  --short_input_len 24 --pred_len 720 --attn_pieces 4 --learning_rate 0.0001 --dropout 0.1 --Sample_strategy Avg --n_heads 4 --sample_train 192 --batch_size 16 --itr 5 --train
