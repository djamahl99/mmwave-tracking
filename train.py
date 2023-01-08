import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import wandb

from datasets.mmwave_tracking import MMWaveTracking
from models.perceiver_io import PerceiverTracking
from models.matcher import HungarianMatcher

from tqdm import tqdm
import itertools

def matched_loss(outs, targets, criterion):
    n = 5
    perms = itertools.permutations(range(n), n)
    loss = 0.0

    for js in perms:
        for i, j in enumerate(js):
            loss += criterion(outs[:, i], targets[:, j])

    return loss

def main():
    ds = MMWaveTracking()
    m = PerceiverTracking(dim=256)

    print(m)

    train(ds, m)

def train(train_ds, model):
    epochs = 100

    # dataset stuff
    trainloader = data.DataLoader(train_ds, batch_size=64, shuffle=True)

    # optimizer = optim.SGD(params=model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.001)
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-4)

    device = torch.device("cpu")
    model = model.to(device)

    matcher = HungarianMatcher()

    bce_criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()

    wandb.init(
        project="mmwave",
        config=dict(
        batch_size =trainloader.batch_size,
        # lr=optimizer.lr,
        model=model._get_name()
    ))

    wandb.watch(models=[model])

    for epoch in range(epochs):
        with tqdm(desc=f"EPOCH {epoch}", unit='time pt') as pbar:
            for pointcloud, objects, scores in tqdm(trainloader, unit='batch'):
                pointcloud = pointcloud.to(device, dtype=torch.float32)
                objects = objects.to(device, dtype=torch.float32)
                scores = scores.to(device, dtype=torch.float32)

                # print(pointcloud.shape, objects.shape, scores.shape)

                model.zero_grad()

                # prediction
                pred_pos, pred_scores = model(pointcloud)

                loss_bce = 0.0
                loss_mse = 0.0

                # bipartite matching loss
                matchings = matcher(outputs={'pred_logits': pred_scores, 'pred_regs': pred_pos}, 
                        targets={'labels': scores, 'regression': objects})

                for b, (i_matched, j_matched) in enumerate(matchings):
                    for k in range(len(i_matched)):
                        i, j = i_matched[k], j_matched[k]
                        loss_bce += bce_criterion(pred_scores[b, i], scores[b, j])
                        loss_mse += mse_criterion(pred_pos[b, i], objects[b, j])

                loss = loss_bce + loss_mse

                loss.backward()
                optimizer.step()
                

                wandb.log({'loss/scores': loss_bce.item(), 'loss/position': loss_mse.item(), 'loss/total': loss.item()})
    

if __name__ == "__main__":
    main()