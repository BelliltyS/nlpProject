import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from nltk.translate.bleu_score import corpus_bleu

from transformers import AutoTokenizer
from seqtoseq import Seq2Seq, TranslationDataSet

def preprocess():

    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
    # model = BertForMaskedLM.from_pretrained("bert-base-german-cased")
    bert_tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
    # Read the train.labeled file and convert to tokenized input-output pairs
    input_ids_per_group = []
    output_ids_per_group = []
    with open("data/train.labeled", "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] == "German:\n":
                input_ids = []
                j = 1
                while lines[i + j] != "English:\n":
                    input_ids.append(tokenizer.encode(lines[i + j].strip(),
                                                      return_tensors='pt'))  # max lenght, enlever virgul et point
                    j += 1
                input_ids_per_group.append(input_ids)
            if lines[i] == "English:\n":
                output_ids = []
                k = 1
                while lines[i + k] != "\n":
                    output_ids.append(bert_tokenizer_en.encode(lines[i + k].strip(), return_tensors='pt'))
                    k += 1
                output_ids_per_group.append(output_ids)

    print(input_ids_per_group[0])
    print(output_ids_per_group[0])

    pairs = []
    # paragraph_pairs = []

    for paragraph_germ, paragraph_en in zip(input_ids_per_group, output_ids_per_group):
        paragraph_pairs = []
        for sen_germ, sen_en in zip(paragraph_germ, paragraph_en):
            paragraph_pairs.append([sen_germ, sen_en])
        pairs.append(paragraph_pairs)
    print("pairs[0]: ", pairs[0])

    #enregistrer dans fichier



if __name__ == "__main__":
    preprocess()
    #sortir des fichiers
    vocab_size_enc = 0
    vocab_size_dec = 0
    hp = dict(hidden_size_enc=128, hidden_size_dec=128, batch_size=256, epochs=10, lr=0.001)

    path_train = ""
    path_eval = ""
    train_ds = TranslationDataSet(path_train)
    eval_ds = TranslationDataSet(path_eval)

    datasets = {"train": train_ds, "test": eval_ds}

    model = Seq2Seq(vocab_size_enc, hp['hidden_size_enc'], vocab_size_dec, hp['hidden_size_dec'], hp['batch_size'])

    # Define the criterion
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])



    # Tokenize the input and output texts
    # input_ids = [tokenizer.encode(text, return_tensors="pt", add_special_tokens=True) for t in input_ids_per_group for text in t ]
    # output_ids = [tokenizer.encode(text, return_tensors="pt", add_special_tokens=True) for text in output_ids_per_group]

    # Create DataLoader objects for the training data

    """
    train_inputs = torch.cat(input_ids)
    train_outputs = torch.cat(output_ids)
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)"""








    for epoch in range(3):
        model.train()
        for i, (input_ids, labels) in enumerate(train_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, masked_lm_labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

    with open("data/comp.unlabeled", "r") as f:
        lines = f.readlines()
        german_paragraphs = []
        current_paragraph = ""
        for line in lines:
            if line != "\n":
                current_paragraph += line
            else:
                german_paragraphs.append(current_paragraph)
                current_paragraph = ""

    translated_paragraphs = []
    for german_paragraph in german_paragraphs:
        input_ids = tokenizer.encode(german_paragraph, return_tensors="pt")
        outputs = model.generate(input_ids)
        translated_paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_paragraphs.append(translated_paragraph)
    with open("comp_id1_id2.labeled", "w") as f:
        for i in range(len(translated_paragraphs)):
            f.write("German:\n")
            f.write(german_paragraphs[i])
            f.write("\n")
            f.write("English:\n")
            f.write(translated_paragraphs[i])
            f.write("\n\n")
    import nltk

    nltk.download('bleu_score')

    reference_file = "val.labeled"
    candidate_file = "comp_id1_id2.labeled"

    with open(reference_file, "r") as ref_f, open(candidate_file, "r") as cand_f:
        reference_text = []
        candidate_text = []
        reference_lines = ref_f.readlines()
        candidate_lines = cand_f.readlines()
        for i in range(len(reference_lines)):
            if reference_lines[i] == "English:\n":
                sentence = []
                j = 1
                while reference_lines[i + j] != "\n":
                    sentence.append([reference_lines[i + j].strip()])
                reference_text.sentence
        if candidate_lines[i] == "English:\n":
            sentence2 = []
            k = 1
            while candidate_lines[i + k] != "\n":
                sentence2.append([candidate_lines[i + j].strip()])

            candidate_text.sentence2

        bleu_score = corpus_bleu(reference_text, candidate_text)
        print("BLEU score: ", bleu_score)

        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] == "German:\n":
                input_ids = []
                j = 1
                while lines[i + j] != "English:\n":
                    input_ids.append(tokenizer.encode(lines[i + j].strip()))
                input_ids_per_group.append(input_ids)
            if lines[i] == "English:\n":
                output_ids = []
                k = 1
                while lines[i + j] != "\n":
                    output_ids.append(tokenizer.encode(lines[i + k].strip()))
                output_ids_per_group.append(output_ids)
