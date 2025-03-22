export DATASET_KEY=$1
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR5="./makeup_assets/${DATASET_KEY}"
export OUTPUT_DIR5="./trained_tokens/${DATASET_KEY}"
export PLACEHOLDER="<${DATASET_KEY}*>"
export INITIALIZER=$2

echo $1 $2

python train_textualinversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR5 \
  --learnable_property="object" \
  --placeholder_token=${PLACEHOLDER} \
  --initializer_token=${INITIALIZER} \
  --num_vectors=5 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=1.0e-5 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR5 \
  --mixed_precision="fp16"