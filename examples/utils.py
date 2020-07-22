import os

import torch
import nltk
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel
from transformers.tokenization_utils import trim_batch
import json
import time
import ast
fi = open('/dccstor/tuhinstor/time.txt','a')
f = open('/dccstor/tuhinstor/tuhin/randarr.json')
for line in f:
    arr = json.loads(line)

mock_pos = arr["array"]

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')
model.cuda()
model.eval()

def getRobertaCLS(sent):
    input_ids = torch.tensor(tokenizer.encode(sent,max_length=512)).unsqueeze(0).cuda()  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs[0]
    CLS = last_hidden_states[0,0].half() #.cuda()    #last_hidden_states.tolist()[0][0]
    return CLS #torch.cuda.HalfTensor(CLS)

def getRobertaSEP(sent):
    input_ids = torch.tensor(tokenizer.encode(sent,max_length=512)).unsqueeze(0).cuda()  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs[0]
    SEP = last_hidden_states[0,-1].half() #.cuda() #last_hidden_states.tolist()[0][-1]
    return SEP #torch.cuda.HalfTensor(SEP)

def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    embeddings = []
    ques = []
    contexts = []
    if 'train.source' in data_path:
        for x,y in zip(open('/dccstor/tuhinstor/tuhin/testamrrob/train_amr.json'),open('/dccstor/tuhinstor/tuhin/testamrrob/train_ques.amr.anonymized')):
            ques.append(y.strip())
            contexts.append(json.loads(x.strip()))

    if 'val.source' in data_path or 'test' in data_path:
        for x,y in zip(open('/dccstor/tuhinstor/tuhin/testamrrob/val_amr.json'),open('/dccstor/tuhinstor/tuhin/testamrrob/val_ques.amr.anonymized')):
            ques.append(y.strip())
            contexts.append(json.loads(x.strip()))

    index = 0
    with open(data_path, "r") as f:
        for text in f.readlines():
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
            sentences = []
            roberta_embeddings = []
            if 'source' in data_path:
                question = text.split(' ----------- ')[0]
                text = text.split(' ----------- ')[1]
                sentences = nltk.sent_tokenize(text)
                sentences.insert(0,question)
                #question = ques[index]
                #sentences = contexts[index]
                #sentences.insert(0,question)
                #fi.write("Writing"+'\t'+str(data_path)+'\t'+str(question)+'\t'+str(question_t)+'\t'+str(len(sentences))+'\n')
                for i in range(49):
                    if i<len(sentences) and (sentences[i]=='' or len(sentences[i])<5):
                        sentences[i]='dummy_string'
                    if i==0:
                        roberta_embeddings.append(getRobertaCLS(sentences[i]))
                        roberta_embeddings.append(getRobertaSEP(sentences[i]))
                    else:
                        if i>=len(sentences):
                            roberta_embeddings.append(torch.cuda.HalfTensor(mock_pos))
                        else:
                            roberta_embeddings.append(getRobertaCLS(sentences[i]))
                roberta_embeddings = torch.stack(roberta_embeddings)
            embeddings.append(roberta_embeddings)
            index = index+1
    return examples,embeddings


class SummarizationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        f = open('/dccstor/tuhinstor/time.txt','a')
        start_time = time.time()
        self.source, self.source_emb = encode_file(tokenizer, os.path.join(data_dir, type_path + ".source"), max_source_length)
        f.write("Preprocessing "+str((time.time() - start_time))+'\n')
        self.target, self.target_emb = encode_file(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        roberta_embeddings = self.source_emb[index]
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, 'roberta': roberta_embeddings}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        rob_emb = []
        rob_emb_new = []
        max_seq_rob = -1  
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        roberta_embeddings = torch.stack([x["roberta"] for x in batch])#torch.stack(rob_emb_new)
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y,"roberta_embeddings": roberta_embeddings}
