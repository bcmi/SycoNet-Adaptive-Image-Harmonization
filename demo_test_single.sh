python demo_test.py \
--name syco \
--checkpoints_dir checkpoints \
--model laBaseLUTs \
--netEr Syco \
--epoch pretrained \
--dataset_mode custom \
--gpu_ids 0 \
--is_train 0 \
--real examples/f436_1_1.jpg \
--mask examples/f436_1_1.png \
--results_dir results \
--augment_num 10 \
--keep_res \
--eval \