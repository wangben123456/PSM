import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import SinusoidalPositionalEmbedding, DualTransformer

class PSM(nn.Module):
    """Simple query-proposal encoder producing global representations."""
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.hidden_dim = config['hidden_dim']
        self.frame_fc = nn.Linear(config['frame_feat_dim'], self.hidden_dim)
        self.word_fc = nn.Linear(config['word_feat_dim'], self.hidden_dim)
        self.pred_vec_v = nn.Parameter(torch.zeros(config['frame_feat_dim']).float(), requires_grad=True)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(self.hidden_dim, 0, config['max_num_words']+1)
        self.query_encoder = DualTransformer(**config['DualTransformer'])
        self.prop_encoder = DualTransformer(**config['DualTransformer'])

    def forward(self, frames_feat, frames_len, words_feat, words_len, **kwargs):
        bsz, n_frames, _ = frames_feat.shape

        pred_vec_v = self.pred_vec_v.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec_v], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len + 1)

        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = words_feat + words_pos
        words_mask = _generate_mask(words_feat, words_len)

        _, q_out = self.query_encoder(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
        query_repr = q_out.mean(dim=1)

        _, p_out = self.prop_encoder(words_feat, words_mask, frames_feat, frames_mask, decoding=1)
        prop_repr = p_out.mean(dim=1)

        return {
            'query_repr': query_repr,
            'prop_repr': prop_repr
        }

def _generate_mask(x, x_len):
    mask = []
    for l in x_len:
        mask.append(torch.zeros([x.size(1)]).byte().cuda())
        mask[-1][:l] = 1
    mask = torch.stack(mask, 0)
    return mask
