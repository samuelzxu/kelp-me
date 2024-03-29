import wandb
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from torchvision.transforms import RandomAffine, InterpolationMode, RandomHorizontalFlip, RandomVerticalFlip, Compose, ColorJitter, GaussianBlur
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerConfig,
    Trainer,
    TrainingArguments
)
from models import SegformerForKelpSemanticSegmentation
import torch
from torch import nn
import evaluate
import os
import fire
from model_configs import config_b1, config_b3

def main(
        output_dir: str,
        model: str,
        hub_model_name: str = 'segformer-b1-from-scratch',
        hf_dataset_identifier: str = "samitizerxu/kelp_data_rgb_swin_nir",
        test_run: bool = False,
        wdb_proj: str = "kelp-segmentation",
        use_wandb : bool = True,
        hf_username: str = 'samitizerxu',
        shuffle_seed: int = 1,
        split_seed: int = 1,
        test_prop: float = 0.2,
        batch_size: int = 8,
        epochs: int = 40,
        lr: float = 0.00002,
):
    wandb.login()
    wandb.init()
    wandb.run.name = hub_model_name
    wandb.run.save()

    os.environ["WANDB_PROJECT"]=wdb_proj
    
    report_to =  None if (not use_wandb or test_run) else 'wandb'

    if test_run:
        ds = load_dataset(hf_dataset_identifier, split='train[:16]')
    else:
        ds = load_dataset(hf_dataset_identifier)

    ds = ds.shuffle(seed=shuffle_seed)
    
    ds = ds["train"].train_test_split(test_size=test_prop, seed=split_seed)
    train_ds = ds["train"]
    test_ds = ds["test"]

    if test_run:
        epochs=1

    torch.cuda.empty_cache()

    processor = SegformerImageProcessor()

    def train_transforms(example_batch):
        state = torch.get_rng_state()
        transform_fn = Compose([
            RandomAffine(degrees=90,translate=(0.3,0.3),scale=(0.7,1.3),interpolation=InterpolationMode.BILINEAR ),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            GaussianBlur(kernel_size=(5,15), sigma=(0.1, 5)),
            ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5))
        ])
        images = [transform_fn(x) for x in example_batch['pixel_values']]
        torch.set_rng_state(state)
        transform_fn = Compose([
            RandomAffine(degrees=90,translate=(0.3,0.3),scale=(0.7,1.3),interpolation=InterpolationMode.BILINEAR ),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])
        labels = [transform_fn(x) for x in example_batch['label']]
        inputs = processor(images, labels)
        return inputs


    def val_transforms(example_batch):
        images = [x for x in example_batch['pixel_values']]
        labels = [x for x in example_batch['label']]
        inputs = processor(images, labels)
        return inputs

    # Set transforms
    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)
    
    if model == 'segformer-b3':
        config = SegformerConfig(
            semantic_loss_ignore_index=255,
            num_channels=3,
            **config_b3
        )
    elif model == 'segformer-b1':
        config = SegformerConfig(
            semantic_loss_ignore_index=255,
            num_channels=3,
            **config_b1
        )
    else:
        raise ValueError('Model config not valid')
    
    model = SegformerForKelpSemanticSegmentation(
        config,
    )

    training_args = TrainingArguments(
        output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=5,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=30,
        eval_steps=30,
        report_to=report_to,
        metric_for_best_model='eval_loss',
        logging_steps=1,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
        push_to_hub=True,
        warmup_ratio=0,
        weight_decay=0.1,
        hub_model_id=hub_model_name,
        hub_strategy="end",
    )

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            # scale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            pred_labels = np.round(torch.nn.functional.sigmoid(logits_tensor.detach().cpu()).numpy())
            print("Pred labels 1 sum: ",(pred_labels==1).sum())
            intsc_sum = (pred_labels.astype(np.uint8) & labels.astype(np.uint8)).sum()
            union_sum = (pred_labels.astype(np.uint8) | labels.astype(np.uint8)).sum()
            metrics = {
                'eval_iou_kelp': intsc_sum / union_sum
            }

            return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    kwargs = {
        "tags": ["vision", "image-segmentation"],
        "dataset": hf_dataset_identifier,
    }

    processor.push_to_hub(hub_model_name) 
    trainer.push_to_hub(**kwargs)  

if __name__ == '__main__':
    fire.Fire(main)