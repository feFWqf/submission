dataset=sims
model=cmfenet
# run_once
python /home/XXX/run_once.py --datasetName $dataset --modelName $model \
   --model_save_dir results/$dataset/$model/run_once/models \
   --res_save_dir results/$dataset/$model/run_once/results \
   --save_model \
   --gpu_ids $1
echo "done!"

