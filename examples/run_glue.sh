export GLUE_DIR='/dccstdor/tuhinstor/tuhin/'

python /dccstor/tuhinstor/transformers/examples/run_glue.py \
  --model_name_or_path roberta-large \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 10 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $GLUE_DIR/mrpc_output/ \
  --fp16
