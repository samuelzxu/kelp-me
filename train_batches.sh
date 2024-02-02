

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

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b1-run4-lr-0.0001 \
                --hub_model_name segformer-b1-from-scratch-run5 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 3 \
                --shuffle_seed 3 \
                --batch_size 22 \
                --lr 0.0001 \
                --epochs 40
