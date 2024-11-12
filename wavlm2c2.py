import numpy as np
import torch

from ctranslate2.converters.transformers import Wav2Vec2Loader, register_loader
from ctranslate2.specs.wav2vec2_spec import Wav2Vec2Spec
from ctranslate2.specs import (
    attention_spec,
    common_spec,
    model_spec,
    transformer_spec,
)


@register_loader("WavLMConfig")
class WavLMLoader(Wav2Vec2Loader):
    @property
    def architecture_name(self):
        return "WavLMForCTC"

    def get_model_spec(self, model):
        """
        feature_extractor -> feature_projection -> encoder -> lm_head
        """
        model.config = model.wavlm.config
        spec = WavLMSpec(
            model.config.num_feat_extract_layers,
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
        )

        # Encoder
        # layer component name matching (no duplications saving)
        for layer in model.wavlm.encoder.layers:
            layer.self_attn = layer.attention
            layer.self_attn_layer_norm = layer.layer_norm
            layer.activation_fn = layer.feed_forward.intermediate_act_fn
            layer.fc1 = layer.feed_forward.intermediate_dense
            layer.fc2 = layer.feed_forward.output_dense

        self.set_encoder(spec.encoder, model, model.config)
        return spec

    def set_config(self, config, model, tokenizer):
        return

    def get_vocabulary(self, model, tokenizer):
        return [chr(i) for i in range(80, 112)] 

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_feature_extractor(self, spec, feature_extractor, conv_bias):
        spec.feat_layer0.conv.weight = feature_extractor.conv_layers[0].conv.weight
        if conv_bias:
            spec.feat_layer0.conv.bias = feature_extractor.conv_layers[0].conv.bias
        self.set_layer_norm(
            spec.feat_layer0.layer_norm, feature_extractor.conv_layers[0].layer_norm
        )
        for spec_layer, module_layer in zip(
            spec.feat_layer, feature_extractor.conv_layers[1:]
        ):
            spec_layer.conv.weight = module_layer.conv.weight
            if conv_bias:
                spec_layer.conv.bias = module_layer.conv.bias
            self.set_layer_norm(spec_layer.layer_norm, module_layer.layer_norm)

    def set_feature_projection(self, spec, feature_projection):
        self.set_layer_norm(spec.fp_layer_norm, feature_projection.layer_norm)
        self.set_linear(spec.fp_projection, feature_projection.projection)

    def set_pos_conv_embed(self, spec, encoder, config):
        # forcing parameters to be set because some transformers version initializes garbage numbers
        # conv parameters are float16 so force float32 for the loading
        encoder.pos_conv_embed.conv.weight.data = (
            encoder.pos_conv_embed.conv.weight.data.float()
        )
        encoder.pos_conv_embed.conv.bias.data = encoder.pos_conv_embed.conv.bias.float()
        for param in encoder.pos_conv_embed.parameters():
            param.data = param.data.float()
        encoder.pos_conv_embed(torch.randn((1, 1, config.hidden_size)))
        spec.pos_conv_embed.conv.weight = encoder.pos_conv_embed.conv.weight
        spec.pos_conv_embed.conv.bias = encoder.pos_conv_embed.conv.bias

    def set_encoder(self, spec, model, config):
        self.set_feature_extractor(spec, model.wavlm.feature_extractor, model.config.conv_bias)
        self.set_feature_projection(spec, model.wavlm.feature_projection)
        self.set_pos_conv_embed(spec, model.wavlm.encoder, config)
        self.set_encoder_layers(spec, model.wavlm.encoder)
        self.set_linear(spec.lm_head, model.lm_head)

    def set_encoder_layers(self, spec, encoder):
        self.set_common_layers(spec, encoder)

        for layer_spec, layer in zip(spec.layer, encoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_common_layers(self, spec, module):
        self.set_layer_norm(spec.layer_norm, module.layer_norm)



class WavLMConfig(model_spec.ModelConfig):
    """Configuration for the WavLM model."""


class WavLMSpec(Wav2Vec2Spec):
    def __init__(self, feat_layers, num_layers, num_heads):
        super().__init__(feat_layers, num_layers, num_heads)
        self.encoder = WavLMEncoderSpec(feat_layers, num_layers, num_heads)

    def get_default_config(self):
        return WavLMConfig()

    def get_vocabulary_size(self):
        return self.encoder.lm_head.weight.shape[0]


class WavLMLayerNormConvLayer(model_spec.LayerSpec):
    def __init__(self):
        self.conv = common_spec.Conv1DSpec()
        self.layer_norm = common_spec.LayerNormSpec()


class WavLMPosEmbedConvLayer(model_spec.LayerSpec):
    def __init__(self):
        self.conv = common_spec.Conv1DSpec()



class WavLMEncoderSpec(model_spec.LayerSpec):
    def __init__(self, feat_layers, num_layers, num_heads):
        self.num_heads = np.dtype("int16").type(num_heads)
        self.feat_layer0 = WavLMLayerNormConvLayer()
        self.feat_layer = [WavLMLayerNormConvLayer() for i in range(feat_layers - 1)]
        self.fp_layer_norm = common_spec.LayerNormSpec()
        self.fp_projection = common_spec.LinearSpec()
        self.pos_conv_embed = WavLMPosEmbedConvLayer()
        self.layer_norm = common_spec.LayerNormSpec()
        self.layer = [
            transformer_spec.TransformerEncoderLayerSpec() for _ in range(num_layers)
        ]
        self.lm_head = common_spec.LinearSpec()
