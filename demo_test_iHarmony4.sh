python demo_test.py \
--name syco \
--checkpoints_dir checkpoints \
--model laBaseLUTs \
--netEr Syco \
--epoch pretrained \
--dataset_mode iHarmony4 \
--gpu_ids 0 \
--is_train 0 \
--preprocess resize \
--norm batch \
--nz 32 \
--dataset_root ./iHarmony4_dataset \
--dataset_name Hday2night \
--results_dir results \
--augment_num 10 \
--keep_res \
--eval \

