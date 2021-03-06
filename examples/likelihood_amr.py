import argparse
import glob
import logging
import os
import time
import json
import torch
from torch.utils.data import DataLoader
import nltk

from lightning_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
os.environ['CUDA_VISIBLE_DEVICES']='0'
training_fl = True

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

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch):
        print(type(batch['source_mask']),batch["source_mask"]) #batch,len(batch["source_ids"].tolist()[0]),len(batch["source_mask"].tolist()[0]),batch["source_mask"],batch["source_ids"])
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]

        return loss

    def likelihood(self,batch_arr,source,source_mask):
        lossarr = []
        minloss = 999.0
        parent = ''
        #print(source,source_mask,len(source.cpu().tolist()[0]))
        for batch in batch_arr:
            pad_token_id = self.tokenizer.pad_token_id
            source_ids, source_mask, y = source, source_mask, batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)
            loss = outputs[0].item()
            # print("Loss for text :",batch['text'],' is ',loss)
            lossarr.append((loss,batch['text'],batch['parent']))
            if loss<minloss:
                minloss = loss
                answer = batch['text']
                parent = batch['parent']
        return lossarr  #answer

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
            default=1024,
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
        ansq = {}

        for line,line1 in zip(open('/dccstor/tuhinstor/tuhin/NQ-rank1/val.source'),open('/dccstor/tuhinstor/tuhin/NQ-rank1/val.target')):
            q = line.split('-----------')[0].rstrip()
            supp = line.split('-----------')[1].lstrip()
            m[q] = line.strip()
            ansq[q] = line1.strip()
        for line in open('/dccstor/tuhinstor/tuhin/newdata/data1/gold_tokens2.jsonl'):
            line = json.loads(line.strip())
            m1[line['q']] = line['cand']
        count = 0
        fw = open('/dccstor/tuhinstor/tuhin/likelihood_ranked_amr_val.txt','w')
        cou = 1
        for line in open('/dccstor/tuhinstor/tuhin/NQ-rank1/val.source'):
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
            source_id = model.tokenizer.batch_encode_plus([source], max_length=1024, return_tensors='pt')['input_ids']
            ans = model.likelihood(batch_arr,source_id.cuda(),torch.cuda.LongTensor([[1]*len(source_id.cpu().tolist()[0])]))
            ans = sorted(ans,key=lambda tup: tup[0])
            countp = {}
            for res in ans:
                if countp[res[2]]<3:
                    if ansq[q] in res[2]:
                        fw.write(str(cou)+'\t'+ansq[q]+'\n')
                        ansq[q] = "garbage"+ansq[q]
                    else:
                        fw.write(str(cou)+'\t'+res[1])
                    if res[2] not in countp:
                        countp[res[2]]=1
                    else:
                        countp[res[2]]= countp[res[2]]+1
            cou = cou+1



            # try:
            #     if ans[-1]=='\n':
            #          f.write(q+' [SEP] '+ans)
            #     else:
            #         f.write(q+' [SEP] '+ans+'\n')
            # except:
            #     print("Failed for ",q)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SummarizationTrainer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
