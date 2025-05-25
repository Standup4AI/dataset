"""
Script to train the sequence labelling models 
"""
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import glob
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import os

# path_data = "/home/user/standup_comedyclub/dataset/"
path_data = "/home/user/data/standup/dataset/"

# path_dump_model = "../dataset/"
path_dump_model = "/home/user/data/standup/dump_models/" # because we have space on data folder

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_preds = [
        [int2label[p] for (p, l) in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(preds, labels)
    ]
    true_labels = [
        [int2label[l] for (p, l) in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(preds, labels)
    ]

    all_true = [lbl for seq in true_labels for lbl in seq]
    all_pred = [lbl for seq in true_preds for lbl in seq]

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, labels=base_labels, average=None
    )
    results = {
        f"precision_{lbl}": p for lbl, p in zip(label2int.keys(), prec)
    }
    results.update({f"recall_{lbl}": r for lbl, r in zip(label2int.keys(), rec)})
    results.update({f"f1_{lbl}": f for lbl, f in zip(label2int.keys(), f1)})
    # Overall metrics (exclude O class)
    # non_o_idxs = [i for i, lbl in enumerate(label2int.keys()) if lbl != "O"]
    results.update({
        "precision": np.mean(prec[1:]),
        "recall":    np.mean(rec[1:]),
        "f1":        np.mean(f1[1:]),
    })

    for k, l in results.items():
        if isinstance(l, np.float64):
            results[k] = np.round(l, 4)

    return results

metric = evaluate.load("seqeval")
def compute_metrics_seqeval(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_preds = [
        [int2label[p] for (p, l) in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(preds, labels)
    ]
    true_labels = [
        [int2label[l] for (p, l) in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(preds, labels)
    ]
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }


def load_sequences_from_dir(dir_pattern, label2int):
    sequences, tags = [], []
    for fp in glob.glob(dir_pattern):
        df = pd.read_csv(fp)
        sequences.append(df["text"].astype(str).tolist())
        tags.append(df["label"].map(label2int).tolist())
    return Dataset.from_dict({"tokens": sequences, "tags": tags})


def tokenize_and_align_batch(examples, tokenizer, label_all_tokens=False):
    tokenized = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, max_length=tokenizer.model_max_length,
        stride=128, return_overflowing_tokens=True,
        return_offsets_mapping=False
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    all_labels = []
    for i, sample_idx in enumerate(sample_map):
        word_ids = tokenized.word_ids(batch_index=i)
        labels = examples["tags"][sample_idx]
        prev = None
        chunk_labels = []
        for wid in word_ids:
            if wid is None:
                chunk_labels.append(-100)
            elif wid != prev:
                chunk_labels.append(labels[wid])
            else:
                chunk_labels.append(labels[wid] if label_all_tokens else -100)
            prev = wid
        all_labels.append(chunk_labels)
    tokenized["labels"] = all_labels
    return tokenized

def save_results(results, args, dataset_str, dataset_type_str, path_dump_csv_results,
                 args_not_saved = ['list_lang_test', 'only_test']):
    """
    Save the results in a csv file, with append mode if the file already exists.
    """
    df = pd.DataFrame(results, index=[0])

    # add all the args in the df
    for arg in vars(args):
        if arg not in args_not_saved:
            df[arg] = getattr(args, arg)
    df['Dataset'] = dataset_str # Train/val/test
    df['Dataset_type'] = dataset_type_str # Filtered/IS

    # df.remove(columns=['list_lang_test', 'only_test'], inplace=True)

    # import pdb; pdb.set_trace()
    # append to the csv file
    if os.path.exists(path_dump_csv_results):
        print("Appending to results csv file")
        df.to_csv(path_dump_csv_results, mode='a', header=False, index=False)
    else:
        print("Creating the results csv file")
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path_dump_model), exist_ok=True)
        df.to_csv(path_dump_csv_results, index=False)

    return 1


def main():
    parser = argparse.ArgumentParser(description="NER training script with configurable options")
    parser.add_argument("--mclass", action="store_true",
                        help="Use multi-class labeling (default binary)")
    parser.add_argument("--lang", type=str, default="es",
                        help="Language code for train/val dataset directories")
    parser.add_argument("--lang_test", type=str, default="",
                        help="Not used anymore... Use list_lang_test")
    parser.add_argument("--list_lang_test", type=str, nargs='+', default=[],
                        help="List of language codes for test dataset directories")
    parser.add_argument("--evaluation_strategy", choices=["epoch", 50],
                        default="epoch", help="Evaluation strategy for Trainer")
    parser.add_argument("--nb_epoch", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--class_weight", default=True,
                        help="Apply class weights in loss computation")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for the optimizer")
    
    parser.add_argument("--train_folder", type=str, default="Filtered",
                        help="Training folder name")
    parser.add_argument("--dev_folder", type=str, default="Filtered",
                        help="Validation folder name")

    parser.add_argument("--test", action="store_true",
                        help="Test the model after training")
    parser.add_argument("--evaluate_val", action="store_true",
                        help="Evaluate the model on dev after training")
    parser.add_argument("--save_model", action="store_true",
                        help="Save the best model after training")   
    parser.add_argument("--only_test", action="store_true",
                        help="Test the model without training")
    args = parser.parse_args()

    path_dump_csv_results = path_dump_model + 'results' + args.mclass*'_mclass' + '.csv'

    if len(args.list_lang_test) == 0:
        args.list_lang_test = [args.lang]

    fn_model = f"{args.lang}_{args.train_folder}_{args.dev_folder}" + '_mclass'*args.mclass

    # Labels
    global base_labels, label2int, int2label, num_labels
    base_labels = ["O", "B", "I", "L", "U"]
    if args.mclass:
        label2int = {l: i for i, l in enumerate(base_labels)}
        int2label = {i: l for l, i in label2int.items()}
        num_labels = len(base_labels)
    else:
        label2int = {l: 1 for l in base_labels}
        label2int["O"] = 0
        base_labels = ["O", "U"]
        int2label = {i: l for i, l in enumerate(base_labels)}
        num_labels = len(base_labels)

    # Data
    train_ds = load_sequences_from_dir(
        path_data + f"{args.lang}/{args.train_folder}/train/*.csv",
        label2int
    )
    valid_ds = load_sequences_from_dir(
        path_data + f"{args.lang}/{args.dev_folder}/val/*.csv",
        label2int
    )

    # Tokenizer + Tokenization
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenized_train = train_ds.map(
        lambda ex: tokenize_and_align_batch(ex, tokenizer),
        batched=True, batch_size=1, remove_columns=["tokens", "tags"]
    )
    tokenized_valid = valid_ds.map(
        lambda ex: tokenize_and_align_batch(ex, tokenizer),
        batched=True, batch_size=1, remove_columns=["tokens", "tags"]
    )

    # Class weights
    class_weights = None
    if args.class_weight:
        all_labels = [lbl for seq in train_ds["tags"] for lbl in seq]
        weights = compute_class_weight("balanced",
                                       classes=np.arange(num_labels), y=all_labels)
        class_weights = torch.tensor(weights, dtype=torch.float)

        def compute_weighted_loss(model, inputs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # flatten logits and labels for loss
            device = inputs["input_ids"].device
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                ignore_index=-100
            )
            loss = loss_fct(
                logits.view(-1, num_labels),
                labels.view(-1)
            )
            return loss, outputs

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                loss, outputs = compute_weighted_loss(model, inputs)
                return (loss, outputs) if return_outputs else loss
    

    # Model, data collator, trainer
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base", num_labels=num_labels
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,
        logging_strategy=args.evaluation_strategy,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.nb_epoch,
        weight_decay=0.01,
        logging_dir=f"./logs/{fn_model}",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
    )

    # Trainer selection
    if args.class_weight:

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    # Only load model, not training
    if args.only_test:
        print(f"Only testing, not training... Loading {fn_model}")
        args.test = args.only_test
        # loading past model and putting it to device...
        device = trainer.model.device
        trainer.model = AutoModelForTokenClassification.from_pretrained(path_dump_model + fn_model)
        trainer.model.to(device)
    else:
        trainer.train()
        if args.save_model:
            print("Saving best model")
            trainer.save_model(path_dump_model + fn_model)

        if args.evaluate_val:
            results = trainer.evaluate()
            # import pdb; pdb.set_trace()
            save_results(results, args, dataset_str = 'Val', dataset_type_str = args.dev_folder, 
                            path_dump_csv_results=path_dump_csv_results)
        
    if args.test:
        for lang_test in args.list_lang_test:
            # just to save it in the csv file
            args.lang_test = lang_test
            # if there is a test set, load it and evaluate
            if os.path.exists(path_data + f"{lang_test}/Filtered/test"):
                
                print(f"Testing on {lang_test}/IS/test")
                test_ds = load_sequences_from_dir(
                    path_data + f"{lang_test}/IS/test/*.csv",
                    label2int
                )
                tokenized_test = test_ds.map(
                    lambda ex: tokenize_and_align_batch(ex, tokenizer),
                    batched=True, batch_size=1, remove_columns=["tokens", "tags"]
                )
                results = trainer.evaluate(eval_dataset = tokenized_test)
                # import pdb; pdb.set_trace()
                # save the results in a csv file
                save_results(results, args, dataset_str = 'Test', dataset_type_str = 'IS', 
                            path_dump_csv_results=path_dump_csv_results)
            
                print(f"Testing on {lang_test}/Filtered/test")
                test_ds = load_sequences_from_dir(
                    path_data + f"{lang_test}/Filtered/test/*.csv",
                    label2int
                )
                tokenized_test = test_ds.map(
                    lambda ex: tokenize_and_align_batch(ex, tokenizer),
                    batched=True, batch_size=1, remove_columns=["tokens", "tags"]
                )
                results = trainer.evaluate(eval_dataset = tokenized_test)
                # save the results in a csv file
                save_results(results, args, dataset_str = 'Test', dataset_type_str = 'Filtered', 
                            path_dump_csv_results=path_dump_csv_results)
                
                if os.path.exists(path_data + f"{lang_test}/Manual/test"):
                    print(f"Testing on {lang_test}/Manual/test")
                    test_ds = load_sequences_from_dir(
                        path_data + f"{lang_test}/Manual/test/*.csv",
                        label2int
                    )
                    tokenized_test = test_ds.map(
                        lambda ex: tokenize_and_align_batch(ex, tokenizer),
                        batched=True, batch_size=1, remove_columns=["tokens", "tags"]
                    )
                    results = trainer.evaluate(eval_dataset = tokenized_test)
                    # save the results in a csv file
                    save_results(results, args, dataset_str = 'Test', dataset_type_str = 'Manual', 
                                path_dump_csv_results=path_dump_csv_results)
                
            else:
                print(f'No test set found for the specified language {lang_test}.')

if __name__ == "__main__":
    main()
    # do the in python rm -rf ./checkpoints using os
    # do the in python rm -rf ./logs using os
    os.system("rm -rf ./checkpoints")
    # os.system("rm -rf ./logs")
