config_b1 = {
    "attention_probs_dropout_prob": 0.0,
    "classifier_dropout_prob": 0.1,
    "decoder_hidden_size": 256,
    "depths": [
      2,
      2,
      2,
      2
    ],
    "downsampling_rates": [
      1,
      4,
      8,
      16
    ],
    "drop_path_rate": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_sizes": [
      64,
      128,
      320,
      512
    ],
    "image_size": 350,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "mlp_ratios": [
      4,
      4,
      4,
      4
    ],
    "model_type": "segformer",
    "num_attention_heads": [
      1,
      2,
      5,
      8
    ],
    "num_encoder_blocks": 4,
    "patch_sizes": [
      7,
      3,
      3,
      3
    ],
    "sr_ratios": [
      8,
      4,
      2,
      1
    ],
    "strides": [
      4,
      2,
      2,
      2
    ],
    "torch_dtype": "float32",
    "transformers_version": "4.12.0.dev0"
}

config_b3 = {
  "architectures": [
    "SegformerForImageClassification"
  ],
  "attention_probs_dropout_prob": 0.0,
  "classifier_dropout_prob": 0.1,
  "decoder_hidden_size": 768,
  "depths": [
    3,
    4,
    18,
    3
  ],
  "downsampling_rates": [
    1,
    4,
    8,
    16
  ],
  "drop_path_rate": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_sizes": [
    64,
    128,
    320,
    512
  ],
  "image_size": 350,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-06,
  "mlp_ratios": [
    4,
    4,
    4,
    4
  ],
  "model_type": "segformer",
  "num_attention_heads": [
    1,
    2,
    5,
    8
  ],
  "num_encoder_blocks": 4,
  "patch_sizes": [
    7,
    3,
    3,
    3
  ],
  "sr_ratios": [
    8,
    4,
    2,
    1
  ],
  "strides": [
    4,
    2,
    2,
    2
  ],
  "torch_dtype": "float32",
  "transformers_version": "4.12.0.dev0"
}