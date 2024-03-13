

model_name=$1
epoch=$2

python evaluation/generate_preds.py --model_name $model_name --threshold 0 --epoch $epoch
python evaluation/evaluate.py --model_name $model_name --epoch $epoch  
