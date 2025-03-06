import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np

class MTIDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=32):
        self.data = pd.read_csv(csv_path)
        print("\nCSV文件基本信息:")
        print(f"总样本数: {len(self.data)}")
        print("\n列名:", self.data.columns.tolist())
        print("\n标签分布:\n", self.data['label'].value_counts())
        print("\n序列长度统计:")
        self.data['seq_length'] = self.data['sequence'].str.len()
        print(self.data['seq_length'].describe())
        print("\n前5个样本:")
        print(self.data.head())
        
        # 随机采样10%的数据
        self.data = self.data.sample(frac=0.1, random_state=42)
        print(f"\n采样后数据量: {len(self.data)}")
        print("采样后标签分布:\n", self.data['label'].value_counts())
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 分离序列和标签
        self.mirna_seqs = []
        self.target_seqs = []
        self.labels = []
        
        for _, row in self.data.iterrows():
            # 提取miRNA和target序列
            sequence = row['sequence']
            if len(sequence) >= 22:  # miRNA通常是22nt
                mirna = sequence[-22:]  # 最后22个字符作为miRNA
                target = sequence[:-22]  # 剩余部分作为target
                self.mirna_seqs.append(mirna)
                self.target_seqs.append(target)
                label = int(row['label'])
                assert label in [0, 1], f"标签必须是0或1，但得到了{label}"
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mirna = self.mirna_seqs[idx]
        target = self.target_seqs[idx]
        label = self.labels[idx]

        # Tokenize sequences
        mirna_tokens = self.tokenizer(mirna,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        target_tokens = self.tokenizer(target,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        # Concatenate sequences
        seq = torch.cat([
            torch.LongTensor(mirna_tokens["input_ids"]),
            torch.LongTensor(target_tokens["input_ids"])
        ])

        return seq, torch.tensor(label)

def train_epoch(model, device, train_loader, optimizer, epoch, loss_fn):
    """Training loop."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.squeeze())
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validate(model, device, val_loader, loss_fn):
    """Validation loop with AUROC."""
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target.squeeze()).item()
            
            # 确保输出和标签的形���正确
            probs = F.softmax(output, dim=1)  # [batch_size, 2]
            pos_probs = probs[:, 1].cpu().numpy()  # 取正类的概率
            target = target.squeeze().cpu().numpy()
            
            # 打印调试信息
            print(f"Batch predictions shape: {pos_probs.shape}")
            print(f"Batch labels shape: {target.shape}")
            print(f"Sample predictions: {pos_probs[:5]}")
            print(f"Sample labels: {target[:5]}")
            
            all_preds.extend(pos_probs)
            all_labels.extend(target)

    val_loss /= len(val_loader)
    
    # 打印最终的预测和标签信息
    print(f"Total predictions: {len(all_preds)}")
    print(f"Total labels: {len(all_labels)}")
    print(f"Unique labels: {np.unique(all_labels)}")
    print(f"Prediction range: [{min(all_preds)}, {max(all_preds)}]")
    
    auroc = roc_auc_score(all_labels, all_preds)
    
    print(f'Validation Loss: {val_loss:.4f}, AUROC: {auroc:.4f}')
    return val_loss, auroc

def run_mti_train(train_csv, val_csv=None, num_epochs=1):
    # Training parameters
    max_length = 32  # max length for each sequence
    batch_size = 32
    learning_rate = 6e-4
    weight_decay = 0.1

    # Model settings
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
    use_head = True
    n_classes = 2
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Load pretrained model
    model = HyenaDNAPreTrainedModel.from_pretrained(
        './checkpoints',
        pretrained_model_name,
        download=True,
        config=backbone_cfg,
        device=device,
        use_head=use_head,
        n_classes=n_classes,
    )

    # Create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'U', 'T', 'N'],  # Add U for RNA
        model_max_length=max_length * 2 + 2,  # Double length for pair
        add_special_tokens=False,
        padding_side='left'
    )

    # Create data loaders
    train_dataset = MTIDataset(train_csv, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_csv:
        val_dataset = MTIDataset(val_csv, tokenizer, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    model.to(device)
    best_auroc = 0
    best_model_path = 'best_mti_model.pt'

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        train_epoch(model, device, train_loader, optimizer, epoch, criterion)
        
        if val_loader:
            val_loss, auroc = validate(model, device, val_loader, criterion)
            
            # Save best model
            if auroc > best_auroc:
                best_auroc = auroc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auroc': auroc,
                }, best_model_path)
                print(f'New best model saved with AUROC: {best_auroc:.4f}')

    print(f'Training completed. Best AUROC: {best_auroc:.4f}')

if __name__ == "__main__":
    run_mti_train('mti_pairs.csv') 