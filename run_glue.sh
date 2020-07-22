export GLUE_DIR='/dccstor/tuhinstor/tuhin'

python /dccstor/tuhinstor/transformers/run_glue.py \
  --model_name_or_path '/dccstor/tuhinstor/tuhin/mrpc_output5/checkpoint-3500/' \
  --task_name MRPC \
  --do_eval \
  --data_dir $GLUE_DIR/newdata/ \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 10 \
  --per_gpu_eval_batch_size 10 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --output_dir $GLUE_DIR/mrpc_output5/ \
  --fp16
