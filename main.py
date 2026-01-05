import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from torch.utils.tensorboard import SummaryWriter
import random

from utils import evaluate_cla, evaluate_reg
from load_data import MTLDataset
from backbone import CNN_HardBottom

script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Default hyperparameters for MTL_HardBottom model')
    parser.add_argument('--data_path', type=str, required=True,help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--data_length', type=int, default=1024, help='Length of data samples')
    parser.add_argument('--normalize', action='store_true', default=True, help='Enable normalization')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of classes for classification')
    parser.add_argument('--num_conditions', type=int, default=12, help='Number of classes for operating condition classification')
    parser.add_argument('--weight', type=float, default=0, help='Weight for the conditional loss')
    parser.add_argument('--model_depth', type=int, default=7, help='Depth of CNN model')
    parser.add_argument('--init_method', type=str, default='kaiming_uniform', 
                        choices=['kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'orthogonal', 'normal'],
                        help='Weight initialization method')
    parser.add_argument('--ADCNN_improve', action='store_true', help='Use adaptive convolution layer for first layer')
    parser.add_argument('--dropout', type=float, default=0.0001813949478971233, help='Dropout rate')
    parser.add_argument('--train_steps', type=int, default=25000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=0.0016627875709001585, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.5656171241637387e-05, help='Weight decay')
    parser.add_argument('--eval_freq', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle training data')
    parser.add_argument('--max_cache_items', type=int, default=20000, help='Maximum cache size for dataloader')
    parser.add_argument('--test', default=False, action='store_true',help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--log_dir', type=str, default=None,help='directory to save to or load from')
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    # parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def custom_collate(batch):
    data = []
    labels = []
    
    for item in batch:
        x = torch.from_numpy(item[0]).float() if isinstance(item[0], np.ndarray) else item[0].float()
        if x.dim() == 1:
            x = x.unsqueeze(0)  
        data.append(x)
        
        current_label = item[1]
        if isinstance(current_label, tuple):
            processed_labels = []
            for label in current_label:
                if isinstance(label, list):
                    y = torch.tensor(label, dtype=torch.float)
                elif isinstance(label, np.ndarray):
                    y = torch.from_numpy(label).float()
                elif isinstance(label, torch.Tensor):
                    y = label
                elif isinstance(label, (int, np.integer)):
                    y = torch.tensor(label, dtype=torch.long)
                else:
                    raise TypeError(f"Unsupported label type in tuple: {type(label)}")
                processed_labels.append(y)
            y = tuple(processed_labels) 
        elif isinstance(current_label, list):
            y = torch.tensor(current_label, dtype=torch.float) 
        elif isinstance(current_label, np.ndarray):
            y = torch.from_numpy(current_label).float()
        elif isinstance(current_label, torch.Tensor): 
            y = current_label
        else:
            raise TypeError(f"Unsupported label type: {type(current_label)}")
            
        labels.append(y)
    
    stacked_data = torch.stack(data)
    
    if labels and isinstance(labels[0], tuple):
        stacked_labels = tuple(
            torch.stack([label[i] for label in labels]).long()
            for i in range(len(labels[0]))
        )
    else:
        stacked_labels = torch.stack(labels).long()
    
    return stacked_data, stacked_labels
    
def train_step(data, labels, model, optim, eval=False, weight=0.):

    predictions = model(data)
    all_losses = model.loss_fun(predictions, labels)
    fault_loss = all_losses[0]
    cond_loss = all_losses[1]
    loss = fault_loss * (1 - weight) + cond_loss * weight
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()

    if isinstance(predictions, tuple):
        detached_predictions = tuple(p.detach() for p in predictions)
    else:
        detached_predictions = predictions.detach()
        
    return detached_predictions, fault_loss.detach(), cond_loss.detach()

    
def main(config):
    set_seed(config.seed)
    
    if config.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif config.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device: ", device)
    
    # Load temporary dataset to get label mappings
    tmp_dataset = MTLDataset(config, batch_type='train', cache=False)
    print("Fault label mapping:", tmp_dataset.fault_label_mapping)
    print("Condition label mapping:", tmp_dataset.condition_label_mapping)
    config.num_conditions = len(tmp_dataset.condition_names)
    # Create model
    model = CNN_HardBottom(
        num_classes=(config.num_classes, config.num_conditions), 
        model_depth=config.model_depth, 
        init_method=config.init_method, 
        dropout_rate=config.dropout,
        use_ADCNN_improve=getattr(config, 'ADCNN_improve', False)
    )
    model.apply_improved_initialization()
    if(config.compile == True):
        try:
            model = torch.compile(model, backend=config.backend)
            print(f"Model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    model.to(device)
    
    log_dir = config.log_dir
    if log_dir is None:
        log_dir = f'./runs/MTL_HardBottom_{config.seed}_normalize_{config.normalize}_depth_{config.model_depth}_ADCNN_{config.ADCNN_improve}_inti_{config.init_method}_lr_{config.lr}_batchsize_{config.batch_size}_dropout_{config.dropout}_weight_{config.weight}'
    print(f'log_dir: {log_dir}')
    
    # tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    start_train_step = 0
    if config.checkpoint_step > -1:
        resume_path = (
            f'{os.path.join(log_dir, "state")}'
            f'{config.checkpoint_step}.pt'
        )
        print(f'Loading checkpoint from {resume_path}')
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['network_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Checkpoint loaded successfully.')
            start_train_step = config.checkpoint_step +1
        else:
            print(f'Checkpoint {resume_path} not found.')
    else:
        print('Checkpoint loading skipped.')
    
    if not config.test:
        train_dataset = MTLDataset(config, batch_type='train')
        train_dataloader = iter(
            DataLoader(
                train_dataset, 
                batch_size=config.batch_size, 
                # num_workers=config.num_workers,
                collate_fn=custom_collate
                )
            )
        val_dataset = MTLDataset(config, batch_type='val')
        val_dataloader =DataLoader(
                val_dataset,     
                batch_size=config.batch_size, 
                # num_workers=config.num_workers,
                collate_fn=custom_collate
                )
        
        lr_scheduler = get_scheduler(
                'constant_with_warmup',optimizer = optimizer,num_warmup_steps = 1500,
                num_training_steps = config.train_steps
            )
                
        total_loss = 0.
        best_val_acc = 0.
        val_f1_at_best_acc = 0.
        best_val_step = 0
        
        current_weight = config.weight
        
        for step in tqdm(range(start_train_step, config.train_steps)):
            try:
                X, (y1,y2) = next(train_dataloader)
            except StopIteration:
                train_dataloader = iter(
                    DataLoader(
                            train_dataset, 
                            batch_size=config.batch_size,
                            # num_workers=config.num_workers,
                            collate_fn=custom_collate
                            ))
                X, (y1,y2) = next(train_dataloader)

            X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
            
            model.train()
            
            _, fault_loss, cond_loss = train_step(X, (y1,y2), model, optimizer, eval=False, weight=current_weight)
            lr_scheduler.step()
            
            total_loss += fault_loss.item() + cond_loss.item() * current_weight
            writer.add_scalar("Fault Loss/train", fault_loss.item(), step)
            writer.add_scalar("Cond Loss/train", cond_loss.item(), step)
            writer.add_scalar("Total Loss/train", total_loss, step)
            writer.add_scalar("Learning Rate/train", optimizer.param_groups[0]['lr'], step)
            writer.add_scalar("Current Weight/train", current_weight, step)
            
            if step % config.eval_freq == 0:
                model.eval()
                fault_preds = []
                cond_preds = []
                fault_labels = []
                cond_labels = []
                with torch.no_grad():
                    for X, (y1,y2) in val_dataloader:
                        X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
                        pred, _,_ = train_step(X, (y1,y2), model, optimizer, eval=True, weight=current_weight)
                        fault_pred, cond_pred = pred
                        fault_preds.append(fault_pred)
                        cond_preds.append(cond_pred)
                        fault_labels.append(y1)
                        cond_labels.append(y2)      
                all_fault_pred = torch.cat(fault_preds, dim=0)
                all_cond_pred = torch.cat(cond_preds, dim=0)
                all_fault_labels = torch.cat(fault_labels, dim=0)
                all_cond_labels = torch.cat(cond_labels, dim=0)
                
                # Evaluate fault classification
                fault_acc, fault_f1 = evaluate_cla(preds=all_fault_pred, labels=all_fault_labels)
                writer.add_scalar("Fault/Accuracy_val", fault_acc, step)
                writer.add_scalar("Fault/F1_val", fault_f1, step)
                print(f"Step {step} Fault - Accuracy: {fault_acc:>0.4f}, F1: {fault_f1:>0.4f}\n")
                # Evaluate condition classification
                cond_acc, cond_f1 = evaluate_cla(preds=all_cond_pred, labels=all_cond_labels)
                writer.add_scalar("Cond/Accuracy_val", cond_acc, step)
                writer.add_scalar("Cond/F1_val", cond_f1, step)
                print(f"Step {step} Condition - Accuracy: {cond_acc:>0.4f}, F1: {cond_f1:>0.4f}\n")
                
                # save model
                torch.save(
                        dict(network_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict()),
                        f'{os.path.join(log_dir, "state")}{step + 1}.pt'
                        )
                # Early stopping based on fault classification accuracy
                if fault_acc > best_val_acc:
                    best_val_acc = fault_acc
                    val_f1_at_best_acc = fault_f1
                    best_val_step = step
                
        print(f"Training completed. Best validation step: {best_val_step + 1}, "
              f"Best validation ACC: {best_val_acc:>0.4f}, F1: {val_f1_at_best_acc:>0.4f}\n")
            
    else:  # test
        model.eval()

        test_dataset = MTLDataset(config, batch_type='test')
        test_dataloader =DataLoader(
                test_dataset, 
                batch_size=config.batch_size, 
                # num_workers=config.num_workers4, 
                collate_fn=custom_collate)
        fault_preds = []
        cond_preds = []
        fault_labels = []
        cond_labels = []
        for X, (y1, y2) in test_dataloader:
            X, y1, y2 = X.to(device), y1.to(device), y2.to(device)
            with torch.no_grad():
                _, fault_loss, cond_loss = train_step(X, (y1, y2), model, optimizer, eval=True, weight=config.weight)
                fault_pred, cond_pred = model(X)
                fault_preds.append(fault_pred.detach())
                cond_preds.append(cond_pred.detach())
                fault_labels.append(y1)
                cond_labels.append(y2)
        
        all_fault_pred = torch.cat(fault_preds, dim=0)
        all_cond_pred = torch.cat(cond_preds, dim=0)
        all_fault_labels = torch.cat(fault_labels, dim=0)
        all_cond_labels = torch.cat(cond_labels, dim=0)
        
        # Evaluate
        fault_acc, fault_f1 = evaluate_cla(preds=all_fault_pred, labels=all_fault_labels)
        cond_acc, cond_f1 = evaluate_cla(preds=all_cond_pred, labels=all_cond_labels)
        print(f"Test Fault - Accuracy: {fault_acc:>0.4f}, F1: {fault_f1:>0.4f}\n")
        print(f"Test Condition - Accuracy: {cond_acc:>0.4f}, F1: {cond_f1:>0.4f}\n")
        # save results to csv
        csv_path = os.path.join(script_dir, f'result.csv')
        with open(csv_path, 'a') as f:
            f.write("seed,model_depth,init_method,ADCNN_improve,dropout,lr,dropout,weight_decay,fault_acc,fault_f1,cond_acc,cond_f1\n")
            f.write(f"{config.seed},{config.model_depth},{config.init_method},{config.ADCNN_improve},{config.dropout},{config.lr},{config.dropout},{config.weight_decay},{fault_acc},{fault_f1},{cond_acc},{cond_f1}\n")
        return
            
            
     
if __name__ == "__main__":

    main(parse_args())


