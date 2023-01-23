import os
import pickle

import torch
import torch.nn as nn
from torch.nn import Sequential

from torch.utils.data import Dataset, DataLoader
from project_evaluate import compute_metrics
"""
In this example, the encoder and decoder are defined as separate PyTorch nn.Module subclasses. The encoder class has an 
embedding layer and an LSTM layer. The decoder class has an embedding layer, an LSTM layer, a fully connected layer and 
a LogSoftmax layer. The final Seq2Seq class combines the encoder and decoder, and defines a forward method that takes in 
encoder input, decoder input and initial hidden and cell state, and returns the decoder
"""
class TranslationDataSet(Dataset):
    def __init__(self, data_path, percentage_of_data=1):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.list_of_sentences = pickle.load(f)
        index = int(len(self.list_of_sentences) * percentage_of_data)
        self.list_of_sentences = self.list_of_sentences[0:index]

    def __getitem__(self, item):
        return self.list_of_sentences[item]

    def __len__(self):
        return len(self.list_of_sentences)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size_enc, hidden_size_enc, vocab_size_dec, hidden_size_dec, batch_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Sequential(nn.Embedding(vocab_size_enc, batch_size),
                                  nn.LSTM(input_size=batch_size, hidden_size=hidden_size_enc))
        self.decoder = Sequential(nn.LSTM(input_size=hidden_size_enc, hidden_size=hidden_size_dec),
                                  nn.Linear(in_features=hidden_size_dec, out_features=vocab_size_dec),
                                  nn.LogSoftmax(dim=1))

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

def train(model, data_sets, optimizer,criterion,bert_tokenizer_en, hp):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=hp['batch_size'], shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=hp['batch_size'], shuffle=False)}
    model.to(device)
    best_blue = 0.0
    checkpoint_path = os.path.join('checkpoints', ''.join([f"{key}{value}" for (key, value) in hp.items()]))
    try:
        os.makedirs(checkpoint_path)
    except:
        print("dir exists :)")
    valid_blue = []
    valid_loss = []

    for epoch in range(hp['epochs']):
        print(f'Epoch {epoch + 1}/{hp["epochs"]}')
        print('-' * 10)
        loss_history_train_epoch = []
        loss_history_valid_epoch = []
        blue_train_epoch = []
        blue_valid_epoch = []
        # acc_valid_epoch = []

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch_idx, batch in enumerate(data_loaders[phase]):
                if phase == 'train':
                    optimizer.zero_grad()
                    y = model(batch[0].to(device), batch[1].to(device), device)
                    #loss =  calculer loss
                    loss = criterion(y.view(-1, hp['vocab_size_dec']), batch[1].view(-1)) #enlever leq views
                    loss.backward()
                    optimizer.step()
                    loss_history_train_epoch.append(loss)

                    pred = bert_tokenizer_en.decoder(y)
                    true_val = bert_tokenizer_en.decoder(batch[1]) #le batch c que l'englais
                    blue = compute_metrics(pred, true_val)
                    blue_train_epoch.append(blue)
                else:
                    with torch.no_grad():
                        y = model(batch[0].to(device), batch[1].to(device), device)
                        #loss =
                        loss = criterion(y.view(-1, hp['vocab_size_dec']), batch[1].view(-1))#enlever leq views
                        loss_history_valid_epoch.append(loss)

                        pred = bert_tokenizer_en.decoder(y)
                        true_val = bert_tokenizer_en.decoder(batch[1])  # le batch c que l'englais
                        blue = compute_metrics(pred, true_val)
                        blue_valid_epoch.append(blue)


            if phase == 'train':
                epoch_loss_train = torch.mean(torch.stack(loss_history_train_epoch))
                epoch_blue_Score_train = mean(blue_train_epoch) # # a caise du batch size
                print(f'{phase.title()} Train Loss: {epoch_loss_train:.4e} Train blue score: {epoch_blue_Score_train}')
            else:
                epoch_loss_valid = torch.mean(torch.stack(loss_history_valid_epoch))
                epoch_blue_Score_valid = mean(blue_valid_epoch)
                # epoch_acc_valid = torch.mean(torch.stack(acc_valid_epoch))
                print(f'{phase.title()} Valid Loss: {epoch_loss_valid:.4e} Valid blue score: {epoch_blue_Score_valid} ')
                # f'Valid acc: {epoch_acc_valid}')
                valid_loss.append(epoch_loss_valid)
                valid_blue.append(epoch_blue_Score_valid)

                if epoch_blue_Score_valid > best_blue:
                    best_blue = epoch_blue_Score_valid
                    if epoch_blue_Score_valid > 30:
                        with open(os.path.join(checkpoint_path, f'model_{best_blue}.pkl'), 'wb') as f:
                            # torch.save(model, f)
                            try:
                                torch.save(model, f, pickle_protocol=4) #esperons pas besoin
                            except:
                                print("cannot save model")
                                pickle.dump(model, f, protocol=4)

    try:
        with open(os.path.join(checkpoint_path, f'blue_{best_blue}.pkl'), 'wb') as f:
            torch.save(valid_blue, f)
        with open(os.path.join(checkpoint_path, f'loss_{best_blue}.pkl'), 'wb') as f:
            torch.save(valid_loss, f)
    except:
        print("pb")

    print(f'Best Validation blue score: {best_blue:4f}')
    return best_blue
"""
# Initialize the encoder and decoder
encoder = Encoder(in_vocab, embedding_size, hidden_size)
decoder = Decoder(out_vocab, embedding_size, hidden_size)

# Initialize the model
model = Seq2Seq(encoder, decoder)

# Define the loss function and the optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())
"""
