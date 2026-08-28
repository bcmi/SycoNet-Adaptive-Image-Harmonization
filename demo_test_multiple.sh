python demo_test.py \
--name syco \
--checkpoints_dir checkpoints \
--model laBaseLUTs \
--netEr Syco \
--epoch pretrained \
--data_mode multiple \
--gpu_ids 0 \
--is_train 0 \
--real examples/real_images/ \
--mask examples/masks/ \
--batch_size 4 \
--results_dir results_multiple \
--augment_num 10 \
--keep_res \
--eval \

