from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from llm_tools_5.utils import create_dataset, tokenize_texts, tokenize_texts_and_labels, get_training_args, get_lora_config, tokenize_docs
from functools import partial
from peft import LoraConfig, get_peft_model, PeftConfig


class MyTrainer:

    def __init__(self, model_name, x_train, x_test, y_train=None, y_test=None, training_args=None,
                 context_length=None,
                 model_output_dir=None,
                 lora_config=None,
                 pretrained_peft_model=None,
                 type="text"):
        if training_args is None:
            training_args = {}
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name
        self.model_output_dir = model_output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto')
        self.context_length = context_length if context_length is not None else list(self.model.parameters())[-1].shape[0]
        self.dataset = create_dataset(x_train, x_test, y_train, y_test)
        if y_test is None or y_train is None:
            if type == "text":
                self.tokenize_func = partial(tokenize_texts, tokenizer=self.tokenizer,
                                             context_length=self.context_length)
            else:
                self.tokenize_func = partial(tokenize_docs, tokenizer=self.tokenizer,
                                             context_length=self.context_length)
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        else:
            self.tokenize_func = partial(tokenize_texts_and_labels, tokenizer=self.tokenizer)
            self.data_collator = None
        self.tokenized_datasets = self.dataset.map(self.tokenize_func, batched=True,
                                                   remove_columns=self.dataset["train"].column_names)
        self.training_args = TrainingArguments(**get_training_args(training_args))
        if pretrained_peft_model is None:
            self.lora_config = LoraConfig(**get_lora_config(lora_config))
        else:
            self.lora_config = PeftConfig.from_pretrained(pretrained_peft_model)
            self.lora_config.inference_mode = False

        self.peft_model = get_peft_model(
            self.model,
            self.lora_config
        )
        if self.data_collator is None:
            self.trainer = Trainer(
                model=self.peft_model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                train_dataset=self.tokenized_datasets["train"],
                eval_dataset=self.tokenized_datasets["test"]
            )
        else:
            self.trainer = Trainer(
                model=self.peft_model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                train_dataset=self.tokenized_datasets["train"],
                eval_dataset=self.tokenized_datasets["test"],
                data_collator=self.data_collator
            )

    def print_number_of_trainable_model_parameters(self):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in self.peft_model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\nPercentage of trainable params: {trainable_model_params / all_model_params * 100}"

    def train(self):
        self.trainer.train()

    def push_to_hub(self):
        self.trainer.push_to_hub()

    def save(self):
        self.trainer.save_model(self.model_output_dir)
