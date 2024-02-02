
from transformers.modeling_outputs import SemanticSegmenterOutput
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch import nn
from transformers import (
    SegformerModel, 
    SegformerDecodeHead, 
    SegformerPreTrainedModel,
)
from .losses import dice_loss
from typing import Optional, Union, Tuple
import torch

class SegformerForKelpSemanticSegmentation(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            # if self.config.num_labels > 2:
            #     loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
            #     loss = loss_fct(upsampled_logits, labels)
            # elif self.config.num_labels == 1:
            # if self.config.num_labels == 1:
            # valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                # ratio = (labels == 0).sum() /  torch.max(torch.tensor([(labels == 1).sum(),1]))
                # loss_fct = BCEWithLogitsLoss(reduction="mean", pos_weight=torch.ones_like(labels).to(device='cuda')*ratio)
                # loss = loss_fct(upsampled_logits.squeeze(1), labels.float())

                # loss = (loss * valid_mask).mean()
                # loss = sigmoid_focal_loss(upsampled_logits.squeeze(1), labels.float(), gamma=2, reduction='mean')
            loss = dice_loss(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )