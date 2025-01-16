import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Structural_Enhancement(batch_weights, batch_features, pivotalNum):
    batch_adj = (batch_weights > 0.001)

    degree_centrality_tensor = batch_weights.sum(dim=-1)
    degree_maxes = degree_centrality_tensor.max(dim=0, keepdim=True).values
    degree_centrality_tensor /= degree_maxes

    var_mat = torch.var(batch_features, dim=1)
    row_maxes = var_mat.max(dim=0, keepdim=True).values
    var_mat /= row_maxes

    scores_mat = (degree_centrality_tensor + var_mat)/2

    cos_similarity_mat = F.cosine_similarity(batch_features.unsqueeze(1), batch_features.unsqueeze(0), dim=-1, eps=1e-8)
    cos_similarity_mat[cos_similarity_mat <= 0.15] = 0
    cos_similarity_mat *= batch_adj.float()

    scores_list = scores_mat.clone()
    adj_mat = batch_weights.clone()
    for i in range(pivotalNum):
        _, max_id = torch.max(scores_list[i], dim=0)
        # scores_list[i] *= (1 - cos_similarity_mat[i][max_id])
        scores_list *= (1 - cos_similarity_mat[max_id]*(cos_similarity_mat[max_id] > 0.5))
        adj_mat[i][max_id][adj_mat[i][max_id] > 0] = 1

    return adj_mat


def get_U(alpha, classes):
    S = torch.sum(alpha, dim=0, keepdim=True) + classes
    E = alpha
    b = E / (S.expand(E.shape))
    u = classes / S
    return b, u


class IntraGNN(nn.Module):
    def __init__(self, dataset, raw_features_dim, hidden_dim, slope, dropout, instance_classes):
        super(IntraGNN, self).__init__()
        self.dataset = dataset
        self.classes = instance_classes
        self.hidden = hidden_dim

        self.leaky_relu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)

        self.w_gnn0 = nn.Parameter(torch.FloatTensor(raw_features_dim, hidden_dim))
        self.w_gnn1 = nn.Parameter(torch.FloatTensor(raw_features_dim, hidden_dim))
        self.w_gnn2 = nn.Parameter(torch.FloatTensor(raw_features_dim, hidden_dim))
        nn.init.kaiming_normal_(self.w_gnn0, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.w_gnn1, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.w_gnn2, nonlinearity='leaky_relu')

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, instance_classes),
            nn.Softplus()
        )

    def forward(self, features, weights, view_flag):
        M = features.shape[0]
        D = features.shape[-1]
        eye = torch.eye(D).to(device)

        aft_feat = torch.zeros(M, self.hidden).to(device)
        single_results = torch.zeros(M, self.classes).to(device)
        bb = torch.zeros(M, self.classes).to(device)
        uu = torch.zeros(M).to(device)

        for i in range(M):
            u0 = 999
            t = 1
            # Algorithm 1 Reliable Structural Enhancement by Feature De-correlation
            while True:
                adj_mat_sampled = Structural_Enhancement(weights[i], features[i], 5*t)
                adj_mat_sampled = 0.5 * adj_mat_sampled + eye

                feat = torch.matmul(features[i], getattr(self, f'w_gnn{view_flag}'))
                feat = torch.matmul(adj_mat_sampled, feat)
                feat = self.dropout(feat)
                feat = self.leaky_relu(feat)
                feat = feat.transpose(0, 1)
                feat = self.global_avg_pool(feat)
                feat = feat.squeeze(-1)
                single_result = self.fc(feat)

                b, u = get_U(single_result, self.classes)
                if u >= u0 or 5*t >= 40:
                    break
                else:
                    u0 = u
                    aft_feat[i] = feat
                    weights[i] = adj_mat_sampled
                    single_results[i] = single_result
                    bb[i] = b
                    uu[i] = u
                    t += 1

        return aft_feat, weights, single_results, bb, uu
