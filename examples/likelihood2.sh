export DATA="/dccstor/tuhinstor/tuhin/NQ-amr-qc"
export OUTPUT_DIR_NAME=bart_sum
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${DATA}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
#export PYTHONPATH="../../":"${PYTHONPATH}"

python /dccstor/tuhinstor/transformers/examples/likelihood2.py \
--data_dir=$DATA \
--model_name_or_path=bart-large \
--learning_rate=3e-5 \
--train_batch_size=5 \
--eval_batch_size=5 \
--output_dir=$OUTPUT_DIR \
--fp16 \
--do_predict  $@
