from model.RSEA_MVGNN_sub import *


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        loss = F.mse_loss(features, centers_batch)
        return loss


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, features, center_loss_fn):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    center_loss = center_loss_fn(features, p)
    total_loss = 1.5 * A + B + center_loss

    return total_loss


class OneLayer(nn.Module):
    def __init__(self, dataset, features, weights,
                 num_views, instance_classes,
                 dropout, slope,
                 lambda_epochs, hidden_dim):
        super(OneLayer, self).__init__()
        self.dataset = dataset
        self.features = nn.Parameter(features, requires_grad=False)
        self.weights = nn.Parameter(weights, requires_grad=False)

        self.num_views = num_views
        self.classes = instance_classes
        self.num_regions = features.shape[-1]

        self.lambda_epochs = lambda_epochs
        self.center_loss_fn = CenterLoss(instance_classes, hidden_dim, device)

        self.intra_gnns = nn.ModuleList()
        for i in range(num_views):
            self.intra_gnns.append(IntraGNN(dataset, self.num_regions, hidden_dim, slope, dropout, self.classes))

        self.inter_gcn = nn.Parameter(torch.FloatTensor(num_views, num_views).to(device))
        nn.init.xavier_uniform_(self.inter_gcn)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, instance_classes),
            nn.Softplus()
        )

    def forward(self, batch_idx, epoch, y):
        y_tensor = torch.from_numpy(y).to(dtype=torch.int64, device=device)

        view_features_list = []
        adj_sampled_list = []
        single_result_list = []

        # intra-graph aggregation
        for i, intra_gnn in enumerate(self.intra_gnns):
            view_features, adj_sampled, single_result, bb, uu = intra_gnn(self.features[batch_idx, i, :, :],
                                                                          self.weights[batch_idx, i, :, :], i)
            view_features_list.append(view_features)
            adj_sampled_list.append(adj_sampled)
            single_result_list.append(single_result)

        # inter graph aggregation
        loss = 0
        evidence = single_result_list
        for v_num in range(len(evidence)):
            loss += ce_loss(y_tensor, evidence[v_num] + 1, self.classes, epoch, self.lambda_epochs, view_features_list[v_num], self.center_loss_fn)

        batch_features = torch.cat([view_features.unsqueeze(0)
                                    for view_features in view_features_list], dim=0).permute(1, 0, 2)
        U_feat = (torch.exp(torch.var(bb, dim=1))/uu).view(-1, 1, 1) * batch_features
        batch_features = torch.matmul(self.inter_gcn, U_feat)
        batch_features = torch.sum(batch_features, dim=1)
        all_result = self.fc(batch_features)

        loss += ce_loss(y_tensor, all_result + 1, self.classes, epoch, self.lambda_epochs, batch_features, self.center_loss_fn)
        loss = torch.mean(loss)

        return loss, batch_features
