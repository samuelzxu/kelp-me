

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b1-run1-lr-0.1 \
                --hub_model_name segformer-b1-from-scratch-run1 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 1 \
                --shuffle_seed 2 \
                --batch_size 22 \
                --lr 0.1 \
                --epochs 40

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b1-run2-lr-0.01 \
                --hub_model_name segformer-b1-from-scratch-run2 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 1 \
                --shuffle_seed 3 \
                --batch_size 22 \
                --lr 0.01 \
                --epochs 40

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b1-run3-lr-0.001 \
                --hub_model_name segformer-b1-from-scratch-run3 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 3 \
                --shuffle_seed 3 \
                --batch_size 22 \
                --lr 0.001 \
                --epochs 40

#Segformer b1 with highest batch size
python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b1-lr-0.0001-cleaned \
                --model 'segformer-b1' \
                --hub_model_name kelp-from-scratch-segformer-b1-lr-0.0001-cleaned \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbagg_swin_nir_int_cleaned \
                --split_seed 3 \
                --shuffle_seed 3 \
                --batch_size 22 \
                --lr 0.0001 \
                --epochs 50

#Segformer B3 with highest batch size
python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b3-lr-0.001 \
                --model 'segformer-b3' \
                --hub_model_name segformer-b3-from-scratch-lr-0.001 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 4 \
                --shuffle_seed 4 \
                --batch_size 7 \
                --lr 0.001 \
                --epochs 40
