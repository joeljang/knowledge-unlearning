from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import pandas as pd
from collections import Counter
import re
import string
import logging
from Datasets import Custom_Dataset

# This file is exactly the same as Neo_Model.py but with training related
# methods all stripped out.
# The reason for this is, Lightning Deepspeed offload has a bug where
# trainer.valid() will attempt to load the optimzer for some reason
# but it doesn't get loaded properly
# By removing all training related methods, Lightning skips the buggy process
# If you plan to not use deepspeed offload, you may remove this file and just use
# Neo_Model.py instead
# Refer following issue
# https://github.com/Lightning-AI/lightning/issues/14993


class NeoValid(pl.LightningModule):
    def __init__(self, hparams):
        super(NeoValid, self).__init__()
        self.mode = hparams.mode

        # Model Initializaion
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path)
        if 'gpt' in hparams.tokenizer_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Different models have different kwargs
        if 'gpt-neo' in hparams.model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                hparams.model_name_or_path,
                resid_dropout=0,
                embed_dropout=0,
                attention_dropout=0,
                pad_token_id=self.tokenizer.eos_token_id)
        elif 'opt' in hparams.model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                hparams.model_name_or_path, dropout=0, attention_dropout=0, activation_dropout=0)
        else:  # GPT2
            self.model = AutoModelForCausalLM.from_pretrained(
                hparams.model_name_or_path,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                pad_token_id=self.tokenizer.eos_token_id)

        self.save_hyperparameters(hparams)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.target_length = self.hparams.target_length if self.hparams.target_length else self.hparams.input_length
        # getting the index of the target set if there is multiple val sets
        self.target_validation_idx = None
        # Flag to check wheter this is the initial validation before training
        self.init_validation = True
        # Dataframe that stores MA & EL for individual examples
        self.valid_df = None

        self.el_n = self.hparams.el_n
        # Main N to check for early stopping
        self.el_n_main = self.hparams.el_n[0]

    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels
        )
        loss, score = outputs[0], outputs[1]
        return loss, score

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        if self.mode == 'general_lm_eval':
            return self.validation_general_lm(batch)
        elif self.mode == 'unlearn':
            if dataloader_idx in [self.target_validation_idx, -1]:
                return self.validation_forget(batch)
            else:
                self.validation_general_lm(batch)
        else:
            raise Exception(
                f'Currently not supporting {self.mode}')

    def validation_forget(
            self,
            batch,
            dataset_name='target'):
        loss_reduced, score = self._step(batch)
        self.log(
            'val_loss',
            loss_reduced,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

        # This function also measures results of individual examples
        # i.e. results that are not reduced along batch dim
        value_dict = {}

        # MA
        preds, labels = self.validation_ma(batch, dataset_name)
        accs = []
        if len(preds.shape) == 1:
            preds = torch.unsqueeze(preds, 0)
            labels = torch.unsqueeze(labels, 0)

        for pred, label in zip(preds, labels):
            try:
                acc = accuracy(pred, label, ignore_index=-100)
                accs.append(acc)
            except IndexError:
                pass
        if accs:
            accs = torch.stack(accs)
        value_dict['acc'] = accs

        # EL
        el = self.validation_el(batch, dataset_name)
        value_dict.update(el)

        # Generate suffix given a fixed prefix for Table 3.
        max_len = self.target_length
        input_ids = batch['source_ids']
        prompt = input_ids[..., :100]
        pred = self.model.generate(prompt, max_length=max_len)[..., 100:]
        value_dict['preds'] = pred

        # Recalculate loss individually
        shift_logits = score[..., :-1, :].contiguous().squeeze()
        shift_labels = batch['target_ids'][..., 1:].contiguous().squeeze()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_no_reduce = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # reduce along sequence only, leave batch
        if len(batch['target_ids'].shape) > 1:
            loss_no_reduce = loss_no_reduce.view(
                batch['target_ids'].shape[0], -1)  # (batch, seq_len)
        else:
            loss_no_reduce = torch.unsqueeze(loss_no_reduce, 0)
        mean_losses = []
        for seq_loss in loss_no_reduce:
            mean_loss = seq_loss[seq_loss != 0].mean()
            mean_losses.append(mean_loss)

        mean_losses = torch.stack(mean_losses)

        value_dict['doc_id'] = batch['doc_id']
        value_dict['loss'] = mean_losses
        return value_dict

    def validation_ma(self, batch, dataset_name):
        input_ids = batch['source_ids']
        max_len = self.target_length

        labels, preds = [], []
        for i in range(1, max_len):
            label = input_ids[..., i]
            prompt = input_ids[..., :i]
            try:
                pred = self.model.generate(prompt, max_length=i + 1)[:, -1]
            except IndexError:  # if batch == 1
                pred = self.model.generate(torch.squeeze(
                    prompt), max_length=i + 1).squeeze()[-1]

            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        preds = torch.stack(preds)
        labels = torch.stack(labels)

        score = accuracy(preds, labels, ignore_index=-100)
        self.log(
            f'{dataset_name}/acc',
            score,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

        # return individual example results for logging
        return torch.t(preds), torch.t(labels)

    def validation_el(self, batch, dataset_name):
        input_ids = batch['source_ids']
        max_len = self.target_length

        batch_size = input_ids.shape[0]
        N = self.el_n
        numerator = {n: [0] * batch_size for n in N}

        for i in reversed(range(1, max_len)):
            label = input_ids[..., i:max_len]
            prompt = input_ids[..., :i]
            pred = self.model.generate(prompt, max_length=max_len)[..., i:]

            for example_idx in range(batch_size):
                p, l = pred[example_idx], label[example_idx]
                # extraction likelihood
                for n in N:
                    p_ngram = self.ngram_of_1D_tensor(p, n)
                    l_ngram = self.ngram_of_1D_tensor(l, n)
                    l_unique = set(l_ngram)
                    p_tp = [i for i in p_ngram if i in l_unique]
                    try:
                        p_acc = len(p_tp) / len(l_ngram)
                        numerator[n][example_idx] += p_acc
                    except ZeroDivisionError:  # n-gram isn't defined
                        pass

        el_score = {n: [0] * batch_size for n in N}
        for n in N:
            for i, _ in enumerate(numerator[n]):
                el_score[n][i] = numerator[n][i] / \
                    (max_len - 1 - (n - 1))

        for n in N:
            self.log(f'{dataset_name}/el_{n}-gram',
                     sum(el_score[n]) / len(el_score[n]),
                     prog_bar=True,
                     logger=True,
                     add_dataloader_idx=False,
                     sync_dist=True)

        # return individual example results for logging
        ret = {}
        for k in el_score.keys():
            ret[f'el_{k}-gram'] = torch.Tensor(el_score[k])
        return ret

    # Measures benchmark tasks
    def validation_general_lm(self, batch):
        task = batch["task"][0]
        task_type = batch["task_type"][0]

        if task_type == 'ppl':
            loss, score = self._step(batch)
            self.log(
                f'{task}/loss',
                loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                add_dataloader_idx=False,
                sync_dist=True)
        elif task_type == 'classification':
            self.classification_verbalizer(
                padding_length=self.hparams.input_length,
                task=task,
                batch=batch,
                choices=batch["choices"],
                answer_index=batch["answer_index"])
        elif task_type == 'completion':
            self.lambada_evaluation(
                padding_length=self.hparams.input_length,
                task='lambada',
                batch=batch)
        elif task_type == 'dialog':
            self.dialog_evaluation(
                padding_length=self.hparams.input_length,
                task=task,
                batch=batch)
        elif task_type == 'target':
            raise Exception(
                f'You are evaluating "target" on "general_lm_eval" mode, rerun with "unlearn" mode')
        else:
            raise Exception(f'Currently, {task_type} not implemented..')

    def classification_verbalizer(
            self, padding_length, task, batch, choices, answer_index):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"]
        batch_size = len(source_ids)
        answer_idx = [-1] * batch_size
        for i in range(batch_size):
            answer_idx[i] = answer_index[i]

        batch_acc = 0

        inps = []
        cont_toks_list = []
        inplens = []

        answers = torch.zeros(batch_size, len(choices), device=self.device)

        for c_idx in range(len(choices)):
            choice_ids = self.tokenizer.batch_encode_plus(
                list(
                    choices[c_idx]),
                max_length=self.hparams.input_length,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                return_tensors="pt")["input_ids"].tolist()
            for i in range(batch_size):
                context_enc = self.get_rid_of_pad(source_ids[i])
                continuation_enc = self.get_rid_of_pad(choice_ids[i])

                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(padding_length):][:-1],
                    dtype=torch.long
                ).to(self.device)
                inplen, = inp.shape
                cont = continuation_enc

                # pad length from seq to padding_length
                inp = torch.cat([
                    inp,  # [seq]
                    # [padding_length - seq]
                    torch.zeros(padding_length - inplen,
                                dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id
                ], dim=0)
                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            multi_logits = F.log_softmax(self._model_call(
                batched_inps), dim=-1)  # [batch, padding_length, vocab]
            cnt = 0
            for logits, inp, inplen, cont_toks \
                    in zip(multi_logits, inps, inplens, cont_toks_list):

                # Slice to original seq length
                contlen = len(cont_toks)
                original_logits = logits

                # [1, seq, vocab]
                logits = logits[inplen - contlen:inplen].unsqueeze(0)
                # Check if per-token argmax is exactly equal to continuation
                cont_toks = torch.tensor(
                    cont_toks, dtype=torch.long).unsqueeze(0).to(
                    self.device)  # [1, seq]

                logits = torch.gather(
                    logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                # Answer: (log prob, is-exact-match)
                loss = -float(logits.sum())
                answers[cnt][c_idx] = loss
                cnt += 1
            inps = []
            cont_toks_list = []
            inplens = []

        answer_idx = torch.Tensor(answer_idx).to(self.device)
        answers = torch.argmin(answers, dim=1)

        batch_acc = int(torch.where(answers == answer_idx, 1, 0).sum())

        batch_acc_avg = batch_acc / batch_size

        self.log(
            f'{task}/acc',
            batch_acc_avg,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

        return

    def lambada_evaluation(self, padding_length, task, batch):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"].tolist()
        batch_size = len(source_ids)
        batch_loss = 0
        batch_acc = 0
        batch_f1 = 0
        inps = []
        cont_toks_list = []
        inplens = []
        for i in range(batch_size):
            if source_ids[i] == target_ids[i]:
                context_enc = source_ids[i][:padding_length - 10]
                continuation_enc = target_ids[i][padding_length - 10:]
            else:
                context_enc = self.get_rid_of_pad(source_ids[i])
                continuation_enc = self.get_rid_of_pad(target_ids[i])

            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):][:-1],
                dtype=torch.long
            ).to(self.device)
            inplen, = inp.shape
            cont = continuation_enc

            # pad length from seq to padding_length
            inp = torch.cat([
                inp,  # [seq]
                torch.zeros(
                    padding_length - inplen,
                    dtype=torch.long).to(
                    inp.device)  # [padding_length - seq]
            ], dim=0)
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        multi_logits = F.log_softmax(
            self._model_call(batched_inps),
            dim=-1).cpu()  # [batch, padding_length, vocab]
        for logits, inp, inplen, cont_toks \
                in zip(multi_logits, inps, inplens, cont_toks_list):

            # Slice to original seq length
            contlen = len(cont_toks)
            original_logits = logits
            # [1, seq, vocab]
            logits = logits[inplen - contlen:inplen].unsqueeze(0)
            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = torch.tensor(
                cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            predicted = self.ids_to_clean_text(greedy_tokens)
            ground_truth = self.ids_to_clean_text(cont_toks)
            em = self.exact_match_score(predicted[0], ground_truth[0])
            f1 = self._f1_score(predicted[0], ground_truth[0])

            logits = torch.gather(
                logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
            # Answer: (log prob, is-exact-match)
            loss = -float(logits.sum())
            if bool(max_equal) or em == 1:
                batch_acc += 1

            batch_loss += loss
            batch_f1 += f1

        batch_loss_avg = batch_loss / batch_size
        batch_acc_avg = batch_acc / batch_size
        batch_f1_avg = batch_f1 / batch_size
        self.log(
            f'{task}/loss',
            batch_loss_avg,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)
        self.log(
            f'{task}/acc',
            batch_acc_avg,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)
        self.log(
            f'{task}/f1',
            batch_f1_avg,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)
        return

    def dialog_evaluation(self, padding_length, task, batch):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"].tolist()
        batch_size = len(source_ids)

        inps, cont_toks_list, inplens = [], [], []
        for i in range(batch_size):
            context_enc = self.get_rid_of_pad(source_ids[i])
            continuation_enc = self.get_rid_of_pad(target_ids[i])

            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):],
                dtype=torch.long
            ).to(self.device)
            inplen, = inp.shape
            cont = continuation_enc

            # pad length from seq to padding_length
            inp = torch.cat([
                inp,  # [seq]
                # [padding_length - seq]
                torch.zeros(padding_length - inplen,
                            dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id
            ], dim=0)
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        multi_logits = self._model_call(batched_inps)  # [batch, padding_length, vocab]
        
        full_logits, full_cont_toks = [], []
        for logits, inp, inplen, cont_toks \
                in zip(multi_logits, inps, inplens, cont_toks_list):

            # Slice to original seq length
            contlen = len(cont_toks)

            if contlen >= padding_length:
                cont_toks = cont_toks[:int(padding_length / 2)]
                contlen = len(cont_toks)

            # [seq, vocab]
            logits = logits[inplen - contlen - 1:inplen - 1]
            # Check if per-token argmax is exactly equal to continuation
            cont_toks = torch.tensor(
                cont_toks, dtype=torch.long).to(self.device)  # [seq]

            assert logits.shape[0] == cont_toks.shape[0]

            full_logits.append(logits)
            full_cont_toks.append(cont_toks)

        full_logits = torch.cat(full_logits)
        full_cont_toks = torch.cat(full_cont_toks)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(full_logits, full_cont_toks)
        
        generate_input = []
        for source_id in source_ids:
            inplen = len(source_id)
            inp = torch.tensor(source_id, dtype=torch.long).to(self.device)
            inp = torch.cat([
                torch.zeros(padding_length - inplen,
                            dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id,
                inp
            ], dim=0)
            generate_input.append(inp.unsqueeze(0))  # [1, padding_length]

        inputs = torch.cat(generate_input, dim=0)
        attention_masks = inputs.ne(self.tokenizer.pad_token_id).long()
        generated_ids = self.model.generate(inputs, attention_mask=attention_masks, max_new_tokens=32)[:, padding_length:]
        generated_text = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        generated_text = [t.split('\nUser ')[0] for t in generated_text]
        target_text = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        # Debugging
        source_text = self.tokenizer.batch_decode(source_ids, skip_special_tokens=True)
        for s, g, t in zip(source_text, generated_text, target_text):
            print('---------------------')
            print(f'[Prefix] {s}')
            print(f'[Ground Truth] {t}')
            print(f'[Generated] {g}')
            print('---------------------')
        
        f1_batched = 0
        for g, t in zip(generated_text, target_text):
            f1_batched += self._f1_score(g, t)

        unigram_f1 = f1_batched / batch_size

        self.log(
            f'{task}/loss',
            loss,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True),
        self.log(
            f'{task}/f1',
            unigram_f1,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

    # Reduce results from gpus to a single dataframe + determine early stopping
    def validation_epoch_end(self, output):
        if self.hparams.mode in ['unlearn']:
            if self.init_validation:
                log_col_name = 'init'
            else:
                log_col_name = f'{self.current_epoch:02d}'

            # reduce all output from gpus
            if len(self.hparams.valid_sets) > 1:
                outputs = self.all_gather(output)[self.target_validation_idx]
            else:
                outputs = self.all_gather(output)
            keys = outputs[0].keys()  # [doc_id, loss, acc, el]
            full_output = {k: [] for k in keys}

            # gather all outputs
            for out in outputs:
                for k in keys:
                    full_output[k].append(torch.flatten(out[k]))

            # refactor into pandas favorable format
            for k in keys:
                full_output[k] = torch.cat(full_output[k])
                full_output[k] = torch.flatten(full_output[k]).cpu().numpy()

            if len(full_output['preds'].shape) == 1:
                full_output['preds'] = self.tokenizer.decode(
                    full_output['preds'])
            else:
                full_output['preds'] = self.tokenizer.batch_decode(
                    full_output['preds'])

            # except for 'doc_id' rename all keys to include the epoch
            for k in list(keys):
                full_output[f'{k}_{log_col_name}'] = full_output.pop(k)
            full_output['doc_id'] = full_output.pop(f'doc_id_{log_col_name}')
            df = pd.DataFrame(full_output)

            # append to the df that stores all results from all ddp processes
            df['doc_id'] = df['doc_id'].astype(int)
            df = df.drop_duplicates(['doc_id'])
            df = df.set_index('doc_id')
            self.valid_df = self.valid_df.combine_first(df)
            self.valid_df = self.valid_df.reindex(self.valid_df_index)

            # check early stopping criteria
            ma = df[f'acc_{log_col_name}'].mean()
            el = df[f'el_{self.el_n_main}-gram_{log_col_name}'].mean()
            if self.current_epoch >= self.hparams.min_train_epochs:
                if ma < self.hparams.ma_threshold and el < self.hparams.el_threshold:
                    logging.info(
                        f'Early Stopping as Forgetting Threshold is reached, {ma=}, {el=}')
                    self.trainer.should_stop = True

    def on_validation_end(self):
        if self.hparams.mode in [
                'unlearn'] and self.init_validation and self.local_rank == 0:
            self.valid_df.to_csv(
                f'outputs/init_{self.hparams.wandb_run_name}.csv')
            self.init_validation = False

    def get_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path, length=None):
        input_length = length if length else self.hparams.input_length
        output_length = length if length else self.hparams.output_length
        dataset = Custom_Dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            valid_subset_path=valid_subset_path,
            type_path=type_path,
            input_length=input_length,
            output_length=output_length,
            args=self.hparams)
        return dataset

    def val_dataloader(self):
        datasets = []
        target_idx = -1
        for i in range(len(self.hparams.valid_sets)):
            dataset = self.hparams.valid_sets[i]
            valid_subset_path = self.hparams.valid_subset_path[i]
            type_path = self.hparams.valid_type_path[i]
            dataset_name = dataset

            length = None
            if type_path == 'target':
                length = self.target_length

            dataset = self.get_dataset(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                valid_subset_path=valid_subset_path,
                type_path=type_path,
                length=length)
            datasets.append(dataset)

        # Setup the dataframe for logging MA and EL of individual examples
        if self.mode in ['unlearn'] and self.valid_df is None:
            target_idx = self.hparams.valid_type_path.index('target')
            self.target_validation_idx = target_idx
            self.valid_df = datasets[target_idx].dataset
            self.valid_df = self.valid_df.set_index('doc_id')
            self.valid_df_index = self.valid_df.index
            # The reference prefix for logging Table 3.
            self.valid_df['prefix'] = self.valid_df['text'].apply(
                lambda x: self.tokenizer.decode(self.tokenizer.encode(x)[:100]))

        dataloaders = []
        for i, dataset in enumerate(datasets):
            if self.mode in ['unlearn'] and i == target_idx:
                # For the unlearning target data, match the eval batch_size to
                # train batch_size
                batch_size = self.hparams.train_batch_size * \
                    self.hparams.gradient_accumulation_steps
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=self.hparams.num_workers,
                        shuffle=False))
            else:
                # For other evaluation datasets
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.hparams.eval_batch_size,
                        num_workers=self.hparams.num_workers,
                        shuffle=False))
        return dataloaders

    # Below are some utils functions

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            res = self.model(inps)
            return res[0][:, :, :]

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        return rid_of_specials(white_space_fix(
            remove_articles(remove_punc(lower(s)))))

    def ngram_of_1D_tensor(self, X, n):
        grams = [tuple(X[i:i + n].tolist()) for i in range(X.shape[0] - n + 1)]
        return grams

    def get_rid_of_pad(self, tokens):
        while tokens[-1] == -100 or tokens[-1] == self.tokenizer.pad_token_id:
            tokens.pop()
        return tokens

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) ==
                   self.normalize_answer(ground_truth))

    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        return self.lmap(str.strip, gen_text)

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def device(self):
        return self._device
