import sys
import torch
from bs4 import BeautifulSoup
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from data.process_2 import clean_text
from model import DetectionModel
from config import Config
from data_loader import WebDataset


def train(model, data_loader, optimizer, criterion, device, len_data):
    model.train()

    total_loss = 0.0
    total_acc = 0
    for data, labels in tqdm(data_loader):
        # print(data)
        # print(labels)
        input_ids = data['input_ids'].squeeze(1).to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = labels.to(device)

        output = model(input_ids, attention_mask)
        optimizer.zero_grad()

        batch_loss = criterion(output, labels)
        total_loss += batch_loss.item()
        batch_loss.backward()

        total_acc += (output.argmax(dim=1) == labels).sum().item()

        optimizer.step()

    epoch_loss = total_loss / len_data
    acc = total_acc / len_data
    return epoch_loss, acc


def evaluate(model, data_loader, criterion, device, len_data):
    model.eval()
    total_loss = 0.0
    total_acc = 0

    with torch.no_grad():
        for data, labels in data_loader:
            input_ids = data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = labels.to(device)

            output = model(input_ids, attention_mask)

            loss = criterion(output, labels)
            total_loss += loss.item()

            total_acc += (output.argmax(dim=1) == labels).sum().item()

    loss = total_loss / len_data
    accuracy = total_acc / len_data
    return loss, accuracy


def test(model, data_loader, device, len_data):
    total_acc = 0
    with torch.no_grad():
        for data, labels in tqdm(data_loader):
            input_ids = data['input_ids'].squeeze(1).to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = labels.to(device)

            output = model(input_ids, attention_mask)
            total_acc += (output.argmax(dim=1) == labels).sum().item()

    acc = total_acc / len_data
    return acc


def predict(content):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DetectionModel()
    model.load_state_dict(torch.load(Config.checkpoints_path))
    model.to(device)

    html = BeautifulSoup(content, 'html.parser')
    text = ''.join(html.stripped_strings)
    text = clean_text(text)[-512:]

    # return text
    if len(text) < 10:
        return False, None, None

    tokenizer = BertTokenizer.from_pretrained(Config.pretrain_path)
    tokenized_text = tokenizer.encode_plus(text, padding='max_length', max_length=Config.max_length,
                                           truncation=True, return_tensors='pt')

    input_ids = tokenized_text['input_ids'].squeeze(1).to(device)
    attention_mask = tokenized_text['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    result = output.argmax(dim=1).item()
    value = output.squeeze(0)[result].item()

    # return output
    return text, result, value


if __name__ == "__main__":
    model = DetectionModel()

    bert_params = list(map(id, model.bert.parameters()))
    base_params = filter(lambda x: id(x) not in bert_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.bert.parameters(), 'lr': Config.bert_lr}
    ], lr=Config.base_lr)

    criterion = nn.CrossEntropyLoss()

    if Config.use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    num_epochs = Config.num_epochs

    if len(sys.argv) < 2:
        print('input train or test')

    elif sys.argv[1] == 'train':
        # TRAIN
        train_dataset = WebDataset(data_path=Config.train_data, label_path=Config.train_label)
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

        val_dataset = WebDataset(data_path=Config.val_data, label_path=Config.val_label)
        val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True)

        print('start training ...')
        for epoch in range(num_epochs):
            epoch_loss, epoch_acc = train(model, train_loader, optimizer, criterion, device, len(train_dataset))
            print('Epoch [{}/{}], Train Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss,
                                                                               epoch_acc))

            eval_loss, eval_accuracy = evaluate(model, val_loader, criterion, device, len(val_dataset))
            print('Val Loss: {:.4f}, Accuracy: {:.4f}'.format(eval_loss, eval_accuracy))

        # 保存模型
        torch.save(model.state_dict(), Config.checkpoints_path)
        print('checkpoint saved')

    elif sys.argv[1] == 'test':
        # TEST
        model.load_state_dict(torch.load(Config.checkpoints_path))

        test_dataset = WebDataset(Config.test_data, Config.test_label)
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True)

        test_acc = test(model, test_loader, device, len(test_dataset))
        print('Test Accuracy: {:.4f}'.format(test_acc))

    else:
        print('invalid argv')
