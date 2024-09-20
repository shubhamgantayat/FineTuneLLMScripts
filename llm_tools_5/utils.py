from datasets import DatasetDict, Dataset
import uuid


def create_dataset(x_train, x_test, y_train, y_test):
    if y_train is None or y_test is None:
        train_set = Dataset.from_dict({"content": x_train})
        test_set = Dataset.from_dict({"content": x_test})
    else:
        train_set = Dataset.from_dict({"content": x_train, "label": y_train})
        test_set = Dataset.from_dict({"content": x_test, "label": y_test})

    dataset = DatasetDict({
        "train": train_set,
        "test": test_set
    })
    return dataset


def tokenize_docs(element, tokenizer, context_length=2048):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


def tokenize_texts(element, tokenizer, context_length=2048):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        element["content"],
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        context_length
    )
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        element["content"],
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs


def tokenize_texts_and_labels(examples, tokenizer):
    tokenized_batch = tokenizer(examples["content"], padding="max_length", truncation=True)
    tokenized_batch['labels'] = tokenizer(examples['label'], padding='max_length', truncation=True)["input_ids"]
    return tokenized_batch


def get_training_args(kwargs):
    default_args = dict(
        output_dir=str(uuid.uuid1()),
        evaluation_strategy="epoch",
        num_train_epochs=1,
        learning_rate=5e-4,
    )
    if kwargs is not None:
        for k, v in kwargs.items():
            default_args[k] = v

    return default_args


def get_lora_config(kwargs):
    default_config = dict(
        r=16, #attention heads
        lora_alpha=32, #alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # set this for CAUSAL LANGUAGE MODELS (like Bloom, LLaMA) or SEQ TO SEQ (like FLAN, T5)
    )
    if kwargs is not None:
        for k, v in kwargs.items():
            default_config[k] = v
    return default_config
