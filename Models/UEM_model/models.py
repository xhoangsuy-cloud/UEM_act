import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.UEM_model.model_components import BertAttention, LinearLayer, \
    TrainablePositionalEncoding, BertCrossAttention

import ipdb
from scipy.optimize import linear_sum_assignment


class UEM_Net(nn.Module):
    def __init__(self, config):
        super(UEM_Net, self).__init__()
        self.config = config

        self.query_word_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                                hidden_size=config.hidden_size,
                                                                dropout=config.input_drop)

        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)


        self.query_word_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                                 dropout=config.input_drop, relu=True)


        self.query_word_encoder = BertAttention(
            edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                  hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                  attention_probs_dropout_prob=config.drop))

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)

        self.frame_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                   hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                   attention_probs_dropout_prob=config.drop, frame_len=128,
                                                   sft_factor=config.sft_factor))

        self.cross_attention = BertCrossAttention(
            edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                  hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                  attention_probs_dropout_prob=config.drop, frame_len=128,
                  sft_factor=config.sft_factor))

        self.weight_token = None

        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        self.reset_parameters()


    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, batch):

        event_mask = batch['video_event']
        query_feat = batch['text_feat']
        query_labels = batch['text_labels']
        query_word_tokens = batch['word_feat']
        query_word_mask = batch['word_mask']
        # print("query_word_tokens min/max:", query_word_tokens.min(), query_word_tokens.max())
        # print("query_word_mask min/max:", query_word_mask.min(), query_word_mask.max())


        video_frame_feature = batch['video_frame_features']
        video_frame_mask = batch['videos_mask']

        # query encoding
        video_query_feat = self.encode_query_word(query_word_tokens, query_word_mask)

        # frame feat encoding
        video_frame_feat = self.encode_context(video_frame_feature, video_frame_mask)

        total_normalized_query_video_similarity, total_query_video_similarity = self.event_refinement(video_frame_feat,video_query_feat, event_mask)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        return [label_dict, total_normalized_query_video_similarity, total_query_video_similarity]

    def encode_query_word(self, query_feat, query_mask):
        # encode_input đã truncate feat và mask
        encoded_query = self.encode_input(
            query_feat,
            query_mask,
            self.query_word_input_proj,
            self.query_word_encoder,
            self.query_word_pos_embed
        )  # (N, Lq, D)

        # truncate query_mask để khớp với encoded_query
        if query_mask is not None:
            max_len = encoded_query.shape[1]
            query_mask = query_mask[:, :max_len]

        # modular queries
        video_query_feat = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query_feat



    def encode_context(self, video_frame_feature, video_mask=None):
        # truncate nếu dài hơn 128
        if video_frame_feature.shape[1] > 128:
            video_frame_feature = video_frame_feature[:, :128, :]
            video_mask = video_mask[:, :128]  # dùng video_mask, không phải video_frame_mask

        # pad nếu ngắn hơn 128
        elif video_frame_feature.shape[1] < 128:
            fix = 128 - video_frame_feature.shape[1]
            temp_feat = 0.0 * video_frame_feature.mean(dim=1, keepdim=True).repeat(1, fix, 1)
            video_frame_feature = torch.cat([video_frame_feature, temp_feat], dim=1)

            temp_mask = 0.0 * video_mask.mean(dim=1, keepdim=True).repeat(1, fix).type_as(video_mask)
            video_mask = torch.cat([video_mask, temp_mask], dim=1)

        # encode input dùng tên đúng
        video_frame_feat = self.encode_input(video_frame_feature, video_mask,
                                            self.frame_input_proj,
                                            self.frame_encoder,
                                            self.frame_pos_embed,
                                            self.weight_token)

        # mask output
        video_frame_feat = torch.where(video_mask.unsqueeze(-1).repeat(1, 1, video_frame_feat.shape[-1]) == 1.0,
                                    video_frame_feat, 0.0 * video_frame_feat)

        return video_frame_feat

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        # Project input
        feat = input_proj_layer(feat)

        # truncate sequence theo max positional embeddings
        max_len = pos_embed_layer.position_embeddings.num_embeddings
        seq_length = min(feat.shape[1], max_len)
        feat = feat[:, :seq_length, :]

        # truncate mask nếu có
        if mask is not None:
            mask = mask[:, :seq_length]
            mask = mask.unsqueeze(1)  # (N, 1, L)

        # positional embedding
        feat = pos_embed_layer(feat)

        # pass vào encoder
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)  # (N, L, D_hidden)
        else:
            return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
        Returns:
            modular_queries: (N, D) hoặc (N, M, D) nếu modular_vector_mapping out_features=M>1
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, M)
        # ensure mask shape match
        if query_mask is not None:
            mask = query_mask.unsqueeze(-1)  # (N, L, 1)
        else:
            mask = torch.ones(encoded_query.shape[:2] + (1,), device=encoded_query.device)

        # apply mask logits and softmax over sequence dimension
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, mask), dim=1)  # (N, L, M)
        
        # weighted sum
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, M, D)
        
        # nếu M=1, squeeze modular dimension an toàn
        if modular_queries.shape[1] == 1:
            modular_queries = modular_queries.squeeze(1)  # (N, D)
        
        return modular_queries



    def event_refinement(self, video_frame_feat, query_feat, event_mask):
        # print("video_frame_feat shape:", video_frame_feat.shape)
        # print("query_feat shape:", query_feat.shape)
        # print("event_mask shape:", event_mask.shape)
        softmax_clip_mask = event_mask.clone()
        softmax_clip_mask[softmax_clip_mask == 0.0] = -1e9
        softmax_clip_mask = F.softmax(softmax_clip_mask, dim=-1)
        video_events = torch.einsum('BFM,BMH->BFH', softmax_clip_mask, video_frame_feat)
        # print("video_events shape:", video_events.shape)

        normalized_video_events = F.normalize(video_events, dim=-1)
        normalized_query_feat = F.normalize(query_feat, dim=-1)
        normalized_frame_feat = F.normalize(video_frame_feat, dim=-1)

        normalize_query_event_similarity = torch.matmul(normalized_video_events, normalized_query_feat.t()).permute(2, 1, 0)
        # print("normalize_query_event_similarity shape:", normalize_query_event_similarity.shape)

        normalize_query_event_similarity[normalize_query_event_similarity == 0.0] = -1e9

        _, event_index = torch.max(normalize_query_event_similarity, dim=1)
        # print("event_index max:", event_index.max(), "min:", event_index.min())

        total_normalized_query_video_similarity = []
        total_query_video_similarity = []
        for i in range(event_index.shape[0]):
            normalized_query_video_similarity, query_video_similarity = self.query_event_cross_attention(event_index[i], event_mask, video_frame_feat, query_feat[i], normalized_frame_feat, normalized_query_feat[i])
            total_normalized_query_video_similarity.append(normalized_query_video_similarity)
            total_query_video_similarity.append(query_video_similarity)

        total_normalized_query_video_similarity = torch.stack(total_normalized_query_video_similarity, dim=0)
        total_query_video_similarity = torch.stack(total_query_video_similarity, dim=0)

        return total_normalized_query_video_similarity, total_query_video_similarity


    def query_event_cross_attention(self, single_event_index, event_mask, video_frame_feat, single_query_feat, normalized_frame_feat, normalized_single_query_feat):
        # print("single_event_index:", single_event_index)
        # print("event_mask shape:", event_mask.shape)
        # print("video_frame_feat shape:", video_frame_feat.shape)
        # print("single_query_feat shape:", single_query_feat.shape)
        # print("normalized_frame_feat shape:", normalized_frame_feat.shape)
        # print("normalized_single_query_feat shape:", normalized_single_query_feat.shape)
        selected_frame_mask = event_mask[torch.arange(event_mask.size(0)), single_event_index]
        query_video_similarity = torch.matmul(normalized_frame_feat, normalized_single_query_feat)
        selected_frame_similarity = query_video_similarity * selected_frame_mask
        selected_frame_similarity[selected_frame_similarity == 0.0] = -1e9
        softmax_frame_similarity = F.softmax(selected_frame_similarity, dim=-1)
        normalized_refine_event_representation = torch.einsum('BFH,BF->BH', normalized_frame_feat, softmax_frame_similarity)
        normalized_refine_query_video_similarity = torch.matmul(normalized_refine_event_representation, normalized_single_query_feat)

        refine_event_representation = torch.einsum('BFH,BF->BH', video_frame_feat, softmax_frame_similarity)
        refine_query_video_similarity = torch.matmul(refine_event_representation, single_query_feat)

        return normalized_refine_query_video_similarity, refine_query_video_similarity
def mask_logits(target, mask):
    mask = mask.float().to(target.device)  # chuyển mask sang float và cùng device với target
    return target * mask + (1 - mask) * (-1e10)