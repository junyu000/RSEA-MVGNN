from toolbox.DataPreprocess import *
from toolbox.EarlyStop import *
from toolbox.Evaluation import *

import torch
import argparse
import numpy as np
import torch.optim as optim
from model.RSEA_MVGNN import OneLayer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ap = argparse.ArgumentParser(description='RSEA_MVGNN')
ap.add_argument('--dataset', default='HIV', help='Dataset name')
ap.add_argument('--num-views', type=int, default=2, help='Number of views.')
ap.add_argument('--instance-classes', type=int, default=2, help='Number of instance types.')

ap.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
ap.add_argument('--batch-size', type=int, default=4, help='Batch size.')

ap.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
ap.add_argument('--weight_decay', type=float, default=0.003, help='The weight decay of the optimizer.')
ap.add_argument('--dropout', type=float, default=0.3, help='Dropout.')
ap.add_argument('--slope', type=float, default=0.05, help='The slope of Leaky Relu')

ap.add_argument('--patience', type=int, default=5, help='Patience.')
ap.add_argument('--repeat', type=int, default=20, help='Repeat the training and testing for N times.')
ap.add_argument('--save-postfix', default='HIV', help='Postfix for the saved model and result.')

ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the feature matrix.')
ap.add_argument('--lambda-epochs', type=int, default=25, help='Denotes the annealing step.')

args = ap.parse_args()


def RSEA_MVGNN():
    print('\nRSEA_MVGNN_{}...'.format(args.dataset))
    feats_tensor, weighted_adjs_tensor, instances_labels, train_val_test_idx = load_data(args.dataset)
    feats_tensor, weighted_adjs_tensor = tensor_enhancement(args.dataset, weighted_adjs_tensor)

    features = torch.tensor(feats_tensor, dtype=torch.float32).to(device)
    weighted_adjs = torch.tensor(weighted_adjs_tensor, dtype=torch.float32).to(device)

    train_idx = train_val_test_idx['train']
    val_idx = train_val_test_idx['val']
    test_idx = train_val_test_idx['test']

    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    kmeans_nmi_lists = []
    kmeans_ari_lists = []

    for rti in range(args.repeat):
        print(f'\n---repeat: {rti}---')
        model = OneLayer(args.dataset, features, weighted_adjs,
                         args.num_views, args.instance_classes,
                         args.dropout, args.slope,
                         args.lambda_epochs, args.hidden_dim)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_stop = early_stopping(patience=args.patience, verbose=True,
                                    save_path=f'checkpoint/checkpoint_{args.save_postfix}.pth')

        def train_step(model, optimizer, batch_idx, batch_labels, epoch):
            optimizer.zero_grad()
            loss, batch_features = model(batch_idx, epoch, batch_labels)
            loss.backward()
            optimizer.step()
            return loss, batch_features

        train_idx_generator = index_generator(batch_size=args.batch_size,indices=train_idx)
        val_idx_generator = index_generator(batch_size=args.batch_size,indices=val_idx,shuffle=False)
        save_loss = 100

        for epoch in range(args.epochs):
            print(f'\n---epoch: {epoch}---')
            model.train()

            # train
            train_num_batchs = train_idx_generator.num_iterations()
            for iter in range(train_num_batchs):
                batch_train_idx = train_idx_generator.next()
                batch_train_labels = instances_labels[batch_train_idx]
                train_loss, _ = train_step(model, optimizer, batch_train_idx, batch_train_labels, epoch)
                if iter == train_num_batchs - 1 or iter == int(train_num_batchs/2):
                    print(f'Epoch {epoch:05d} | Iteration {iter:05d} | Train_Loss {train_loss.item()} ')

            model.eval()
            val_losses = []
            val_num_batchs = val_idx_generator.num_iterations()
            first_loss = 100

            # eval
            for iter in range(val_num_batchs):
                batch_val_idx = val_idx_generator.next()
                batch_val_labels = instances_labels[batch_val_idx]

                with torch.no_grad():
                    val_loss, batch_features = model(batch_val_idx, epoch, batch_val_labels)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Epoch {epoch:05d} | Average Val Loss: {avg_val_loss:.4f}')

            if save_loss > avg_val_loss and first_loss > val_losses[0]:
                save_loss = avg_val_loss

            model_stop(avg_val_loss, val_losses[0], model)
            if model_stop.early_stop:
                print('Early stopping!')
                break

        model.eval()
        model.load_state_dict(torch.load(f'checkpoint/checkpoint_{args.save_postfix}.pth'))
        with torch.no_grad():
            _, batch_feat = model(test_idx, 0, instances_labels[test_idx])

        print(f'save_loss = {save_loss}')

        # evaluate
        svm_macro_f1_list, svm_micro_f1_list, nmi_list, ari_list = evaluate_results_nc(
            batch_feat.cpu().numpy(), instances_labels[test_idx], args.instance_classes)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        kmeans_nmi_lists.append(nmi_list)
        kmeans_ari_lists.append(ari_list)

    svm_macro_f1_lists = np.array(svm_macro_f1_lists).transpose()
    svm_micro_f1_lists = np.array(svm_micro_f1_lists).transpose()
    kmeans_nmi_lists = np.array(kmeans_nmi_lists).transpose()
    kmeans_ari_lists = np.array(kmeans_ari_lists).transpose()

    print('----------------------------------------------------------------')
    print('SVM:Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(np.mean(macro_f1), np.std(macro_f1), train_size)
        for macro_f1, train_size in zip(svm_macro_f1_lists, [0.6, 0.2])]))
    print('SVM:Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(np.mean(micro_f1), np.std(micro_f1), train_size)
        for micro_f1, train_size in zip(svm_micro_f1_lists, [0.6, 0.2])]))
    print('K-means:NMI: ' + '{:.6f}~{:.6f}'.format(np.mean(kmeans_nmi_lists), np.std(kmeans_nmi_lists)))
    print('K-means:ARI: ' + '{:.6f}~{:.6f}'.format(np.mean(kmeans_ari_lists), np.std(kmeans_ari_lists)))
    print(
        f'epochs:{args.epochs} batch:{args.batch_size} ptn:{args.patience} rpt:{args.repeat}\n'
        f'lr:{args.lr} Decay:{args.weight_decay} drop: {args.dropout} slp: {args.slope}\n'
        f'lbd:{args.lambda_epochs} dim:{args.hidden_dim}')


if __name__ == '__main__':
    args.dataset = 'HIV'
    args.num_views = 2
    args.instance_classes = 2

    args.save_postfix = 'RSEA_MVGNN_' + args.dataset

    RSEA_MVGNN()
