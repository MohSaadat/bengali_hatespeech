from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import argparse
from utils import *
from model import Classifier
from datetime import datetime as dt
from tqdm import tqdm

DEFAULT_DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT_BACKBONE = "shihab17/bangla-sentence-transformer"
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 50
DEFAULT_START_LR = 1e-4
DEFAULT_LR_DECAY = 0.01
DEFAULT_SAVE_EVERY = 10

MODEL_SAVE_DIR = "saved_models"

def train_n_validate(corpus,
                    backbone=DEFAULT_BACKBONE,
                    validation_split=DEFAULT_VALIDATION_SPLIT,
                    batch_size=DEFAULT_BATCH_SIZE,
                    num_epochs=DEFAULT_NUM_EPOCHS,
                    device=DEFAULT_DEVICE,
                    start_lr=DEFAULT_START_LR,
                    lr_decay=DEFAULT_LR_DECAY,
                    save_every=DEFAULT_SAVE_EVERY):
    # Create the classifier
    classifier = Classifier(
                            backbone_model_name=backbone,
                            device=device
                            )

    # Create the dataset
    dataset = MyDataSet(corpus)

    # Set the split sizes (80% train, 20% validation)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the loss function and optimizer
    loss_func = nn.BCELoss()
    trainables = classifier.get_all_trainables()
    optimizer = optim.AdamW(trainables, lr=start_lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.00-lr_decay)

    currtime = [str(dt.now().day), str(dt.now().month), str(dt.now().year), str(dt.now().hour), str(dt.now().minute)]

    # create the summary writer to track progress
    writer = SummaryWriter(f"runs/run_{'_'.join(currtime)}")

    # model save directory
    model_save_dir = os.path.join(MODEL_SAVE_DIR, f"model_{'_'.join(currtime)}")
    os.makedirs(model_save_dir)

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_single_epoch(classifier, train_loader, loss_func, optimizer)
        val_loss, val_acc = validate_single_epoch(classifier, val_loader, loss_func)
        lr_scheduler.step()

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        if not ((epoch+1) % save_every):
            torch.save(
                {
                "epoch"                 :   epoch,
                "model_state_dict"      :   model.state_dict(),
                "optimizer_state_dict"  :   optimizer.state_dict()
                },
                os.path.join(model_save_dir, f'classifier_ep{n_epochs}.pth'))

    torch.save(
            {
            "epoch"                 :   epoch,
            "model_state_dict"      :   model.state_dict(),
            "optimizer_state_dict"  :   optimizer.state_dict()
            },
            os.path.join(model_save_dir, 'classifier_final.pth'))
    writer.close()

def train_single_epoch(model, dataloader, loss_func, optimizer):
    model.set_trainable(True)

    loss_in_epoch = 0
    for batch in dataloader:
        curr_sentences = batch['sentence']
        curr_labels = nn.functional.one_hot(batch['label'], num_classes=2).to(model.device)

        curr_outputs = model(curr_sentences)
        optimizer.zero_grad()
        curr_loss = loss_func(curr_outputs, curr_labels.to(curr_outputs.dtype))
        curr_loss.backward()
        optimizer.step()

        loss_in_epoch += curr_loss.item()

    return loss_in_epoch/len(dataloader)

def validate_single_epoch(model, dataloader, loss_func):
    model.set_trainable(False)

    loss_in_epoch = 0
    n_correct_preds = 0
    with torch.no_grad():
        for batch in dataloader:
            curr_sentences = batch['sentence']
            curr_labels = nn.functional.one_hot(batch['label'], num_classes=2).to(model.device)

            curr_outputs = model(curr_sentences)
            curr_loss = loss_func(curr_outputs, curr_labels.to(curr_outputs.dtype))

            loss_in_epoch += curr_loss.item()

            _, preds = torch.max(curr_outputs, dim=1)
            n_correct_preds += (preds == curr_labels).sum().item()

    return loss_in_epoch / len(dataloader), n_correct_preds / len(dataloader)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', type=str, default="bhs_naurosromim", help='Dataset corpus [Default: bhs_naurosromim]')
    parser.add_argument('-i', '--inference', type=str, default=None, help='Assign a valid filename with test sentences, \
                                                                        or assign None to train [Default: None, i.e. train]')
    parser.add_argument('-b', '--backbone_model', type=str, default=DEFAULT_BACKBONE, help='Name of the backbone \
                                                                                            (transformer or sentence-transformer) model')
    parser.add_argument('-v', '--validation_split', type=float, default=DEFAULT_VALIDATION_SPLIT, help='Validation split as a fraction')
    parser.add_argument('-b_s', '--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('-e', '--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Number of epochs to train')
    parser.add_argument('-d', '--device', type=int, default=DEFAULT_DEVICE, help=f'Device to run with: -1 for cpu, \
                                                                                positive integer indicates gpu core [Default: {DEFAULT_DEVICE}]')
    parser.add_argument('-l', '--start_lr', type=float, default=DEFAULT_START_LR, help=f'Initial learning rate')
    parser.add_argument('-l_d', '--lr_decay', type=float, default=DEFAULT_LR_DECAY, help=f'Rate of learning rate decay per epoch')
    parser.add_argument('-s', '--save_every', type=int, default=DEFAULT_SAVE_EVERY, help=f'Weights will be saved \
                                                                                                every this number of epochs')
    args = parser.parse_args()

    if args.inference:
        pass
    else:
        train_n_validate(args.corpus,
                        backbone=args.backbone_model,
                        validation_split=args.validation_split,
                        batch_size=args.batch_size,
                        num_epochs=args.num_epochs,
                        device=args.device,
                        start_lr=args.start_lr,
                        lr_decay=args.lr_decay,
                        save_every=args.save_every)
