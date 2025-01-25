import torch
import pandas as pd
import re
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = NewsDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2)

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_samples = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    train_loss = np.mean(losses)
    train_acc = correct_predictions.double() / total_samples
    return train_loss, train_acc

def main():
    
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')

    
    fake['label'] = 0  # 0 for fake news
    true['label'] = 1  # 1 for true (real) news

    data = pd.concat([fake, true], ignore_index=True)

    
    data['text'] = data['text'].apply(clean_text)

    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

   
    train_data.to_csv('train.csv', index=False)
    val_data.to_csv('val.csv', index=False)

    
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

   
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')


    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('val.csv')

    train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(val_data, tokenizer, MAX_LEN, BATCH_SIZE)


    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses = []
    train_accuracies = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler
        )
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

    # Save the trained model
    model_path = "./MODEL"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Model saved to {model_path}")

    # Plotting training metrics
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
