
base = {
	"base": {
		"extractor_mode": "group_norm",
		"extractor_conv_layer_config": [
			(512, 10, 5),
			(512, 3, 2),
			(512, 3, 2),
			(512, 3, 2),
			(512, 3, 2),
			(512, 2, 2),
			(512, 2, 2),
		],
		"extractor_conv_bias": False,
		"encoder_embed_dim": 768,
		"encoder_projection_dropout": 0.1,
		"encoder_pos_conv_kernel": 128,
		"encoder_pos_conv_groups": 16,
		"encoder_num_layers": 12,
		"encoder_num_heads": 12,
		"encoder_attention_dropout": 0.1,
		"encoder_ff_interm_features": 3072,
		"encoder_ff_interm_dropout": 0.0,
		"encoder_dropout": 0.1,
		"encoder_layer_norm_first": False,
		"encoder_layer_drop": 0.05,
		"aux_num_out": None,
	},
}

pretrainer_config = {
	"base": {
		"proj_codevector_dim": 256,  # f
		"codevector_dim": 256,  # d
		"num_codevector_groups": 2,  # G
		"num_codevectors_per_group": 320,  # V
		"gumbel_softmax_temperature": 2.0,  # tau
		"quantizer_dropout": 0.0,
		"mask_time_prob": 0.65,  # p
		"mask_time_length": 10,  # M
		"mask_time_min_masks": 2,
		"gumbel_temperature_decay": 0.999995,
		"min_gumbel_temperature": 0.5,  # tau_min
		"num_negatives": 100,  # K
		"contrastive_loss_temperature": 0.1,  # kappa
		"diversity_loss_weight": 0.1,  # alpha
	},
}

