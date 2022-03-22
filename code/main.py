import argparse

import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import MGCAT_Dataset
from model import MGCAT_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = MGCAT_Dataset(dataset_name=args.dataset, fold=args.fold)
    train_loader = dataset.fetch_train_loader()
    test_loader = dataset.fetch_test_loader()
    model = MGCAT_Model(dataset.features, dataset.edge_index).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        y_pred_train, y_label_train = [], []
        for i, (indexes, labels) in enumerate(train_loader):
            indexes = [indexes[0].to(device), indexes[1].to(device)]
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()
            output, loss_train = model(indexes, labels)
            loss_train.backward()
            optimizer.step()

            label_ids = labels.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 50 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        roc_train, prc_train = roc_auc_score(y_label_train, y_pred_train), average_precision_score(y_label_train,
                                                                                                   y_pred_train)
        print(f'*** Train epoch: {epoch} ***, AUC: {roc_train}, AUPR: {prc_train}')

        test(model, test_loader, epoch)


def test(model, test_loader, epoch):
    model.eval()

    y_pred_test, y_label_test = [], []
    for i, (indexes, labels) in enumerate(test_loader):
        indexes = [indexes[0].to(device), indexes[1].to(device)]
        labels = labels.to(device)
        output, _ = model(indexes, labels)

        label_ids = labels.to('cpu').numpy()
        y_label_test = y_label_test + label_ids.flatten().tolist()
        y_pred_test = y_pred_test + output.flatten().tolist()

    roc_test, prc_test = roc_auc_score(y_label_test, y_pred_test), average_precision_score(y_label_test, y_pred_test)
    print(f'*** Test  epoch: {epoch} ***, AUC: {roc_test}, AUPR: {prc_test}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--dataset', type=str, default='DB2', help='choose which dataset to use, DB1 or DB2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--embedding_size', type=int, default=128, help='64 for DB1, and 128 for DB2')
    parser.add_argument('--fold', type=int, default=1, help='fold')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    default_args = parser.parse_args()

    main(default_args)
