import argparse
import glob
import logging
import os
import time
import json
import torch
from torch.utils.data import DataLoader
import nltk
from transformers import RobertaTokenizer, RobertaModel


from lightning_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
os.environ['CUDA_VISIBLE_DEVICES']='0'

f = open('/dccstor/tuhinstor/tuhin/randarr.json')
for line in f:
    arr = json.loads(line)

mock_pos = arr["array"]

ques = []
contexts = []
for x,y in zip(open('/dccstor/tuhinstor/transition-amr-parser/val_amr.json'),open('/dccstor/tuhinstor/transition-amr-parser/amr_simplifier/val_ques.amr.anonymized')):
    ques.append(y.strip())
    contexts.append(json.loads(x.strip()))

robtokenizer = RobertaTokenizer.from_pretrained('roberta-large')
robmodel = RobertaModel.from_pretrained('roberta-large')
robmodel.cuda()
robmodel.eval()

try:
    from .utils import SummarizationDataset
except ImportError:
    from utils import SummarizationDataset


logger = logging.getLogger(__name__)


class SummarizationTrainer(BaseTransformer):

    mode = "language-modeling"

    def __init__(self, hparams):
        super().__init__(hparams, num_labels=None, mode=self.mode)
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
        )

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None, roberta_emb=None,rob_size=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        roberta_emb=roberta_emb, rob_size=None)


    def getRobertaCLS(self,sent):
        input_ids = torch.tensor(robtokenizer.encode(sent,max_length=512)).unsqueeze(0).cuda()  # Batch size 1
        with torch.no_grad():
            outputs = robmodel(input_ids)
        last_hidden_states = outputs[0]
        CLS = last_hidden_states[0,0] #.half() #.cuda()    #last_hidden_states.tolist()[0][0]
        return CLS #torch.cuda.HalfTensor(CLS)

    def getRobertaSEP(self,sent):
        input_ids = torch.tensor(robtokenizer.encode(sent,max_length=512)).unsqueeze(0).cuda()  # Batch size 1
        with torch.no_grad():
            outputs = robmodel(input_ids)
        last_hidden_states = outputs[0]
        SEP = last_hidden_states[0,-1] #.half() #.cuda() #last_hidden_states.tolist()[0][-1]
        return SEP

    def getRobertaEmbeddings(self,text,index=-1):
        sentences = []
        roberta_embeddings = []
        question = ques[index] #text.split(' ----------- ')[0]
        sentences  = contexts[index] #text.split(' ----------- ')[1]
        #sent = nltk.sent_tokenize(text)
        sentences.insert(0,question)
        #sentences.append(sent)
        for i in range(49):
            if i<len(sentences) and sentences[i]=='':
                sentences[i]='dummy_string'
            if i==0:
                roberta_embeddings.append(self.getRobertaCLS(sentences[i]))
                roberta_embeddings.append(self.getRobertaSEP(sentences[i]))
            else:
                if i>=len(sentences):
                    roberta_embeddings.append(torch.cuda.FloatTensor(mock_pos))
                else:
                    roberta_embeddings.append(self.getRobertaCLS(sentences[i]))
        roberta_embeddings = [torch.stack(roberta_embeddings)]
        return torch.stack(roberta_embeddings)

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]

        return loss

    def likelihood(self,batch_arr,source,roberta_emb,source_mask):
        minloss = 999.0
        parent = ''
        for batch in batch_arr:
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = source, source_mask, batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,roberta_emb=roberta_emb)
            loss = outputs[0].item()
            if loss<minloss:
                minloss = loss
                answer = batch['text']
                parent = batch['parent']
        return parent

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=256,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)

        return {"val_loss": loss, "preds": preds, "target": target}

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+") as p_writer, open(output_test_targets_file, "w+") as t_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
                t_writer.writelines(s + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        return self.test_end(outputs)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = SummarizationDataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle,num_workers=8)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def getanswer(doc,span):
        start = span[0]
        end = span[1]
        arr = []
        for i in range(start,end+1):
            k = str(i)
            if not doc[k].startswith('<'):
                arr.append(doc[k])
        if len(arr)==0:
            s = ''
            return s
        s = ' '.join(arr)
        return s

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        # Add BART specific options
        parser.add_argument(
            "--max_source_length",
            default=974,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )
        return parser


def main(args):

    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)
    model = SummarizationTrainer(args)
    trainer = generic_train(model, args)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        # checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        # model = model.load_from_checkpoint(checkpoints[-1])
        # trainer.test(model)
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1]).cuda()
        m = {}
        m1 = {}
        m2 = {}

        for line in open('/dccstor/tuhinstor/tuhin/NQ-rank/val.source'):
            q = line.split('-----------')[0].rstrip()
            supp = line.split('-----------')[1].lstrip()
            m[q] = line.strip()
        for line in open('/dccstor/tuhinstor/tuhin/newdata/data1/gold_tokens2.jsonl'):
            line = json.loads(line.strip())
            m1[line['q']] = line['cand']
        count = 0
        f = open('/dccstor/tuhinstor/tuhin/testamrrob/likelihood_roberta_bart_amr.txt','w')
        index = 0
        for line in open('/dccstor/tuhinstor/tuhin/NQ-rank/val.source'):
            q = line.split('-----------')[0].rstrip()
            source = m[q]
            logger.info("Doing "+q)
            la_candidates = []
            cand = []
            batch_arr = []
            for c in m1[q]:
                cand = c[2]
                if cand=='':
                    continue
                for cand_sent in nltk.sent_tokenize(cand): # ,
                    batch_arr.append({'text': cand_sent, 'parent': cand,'target_ids': model.tokenizer.batch_encode_plus([cand_sent], max_length=1024, return_tensors='pt')['input_ids'].cuda()})
            source_id = model.tokenizer.batch_encode_plus([source], max_length=974, return_tensors='pt')['input_ids']
            roberta_emb = model.getRobertaEmbeddings(source,index)
            ans = model.likelihood(batch_arr,source_id.cuda(),roberta_emb,torch.cuda.LongTensor([[1]*len(source_id.cpu().tolist()[0])]))
            index = index+1
            try:
                if ans[-1]=='\n':
                     f.write(q+' [SEP] '+ans)
                else:
                    f.write(q+' [SEP] '+ans+'\n')
            except:
                print("Failed for ",q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SummarizationTrainer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
