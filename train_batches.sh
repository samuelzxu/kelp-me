python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b-run0 \
                --hub_model_name segformer-b1-from-scratch-run0 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 1 \
                --shuffle_seed 1 \
                --batch_size 16

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b-run1 \
                --hub_model_name segformer-b1-from-scratch-run1 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 1 \
                --shuffle_seed 2 \
                --batch_size 16

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b-run2 \
                --hub_model_name segformer-b1-from-scratch-run2 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 1 \
                --shuffle_seed 3 \
                --batch_size 16

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b-run3 \
                --hub_model_name segformer-b1-from-scratch-run3 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 3 \
                --shuffle_seed 3 \
                --batch_size 16

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b-run4-80e \
                --hub_model_name segformer-b1-from-scratch-run4 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 3 \
                --shuffle_seed 3 \
                --batch_size 16 \
                --test_prop 0.3 \
                --epochs 80

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b-run5-80e \
                --hub_model_name segformer-b1-from-scratch-run5 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 3 \
                --shuffle_seed 3 \
                --batch_size 16 \
                --test_prop 0.2 \
                --epochs 80

python train_scratch.py --output_dir checkpoints/from-scratch-segformer-b-run6-80e \
                --hub_model_name segformer-b1-from-scratch-run6 \
                --hf_dataset_identifier samitizerxu/kelp_data_rgbaa_swin_nir \
                --split_seed 3 \
                --shuffle_seed 4 \
                --batch_size 16 \
                --test_prop 0.1 \
                --epochs 80