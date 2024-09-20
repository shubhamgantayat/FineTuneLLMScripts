from sklearn.model_selection import train_test_split
from llm_tools_5.peft_fine_tune import MyTrainer as finetune_trainer
import json
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("meta-math/MetaMathQA")
df = dataset["train"].to_pandas()
df["text"] = df[["original_question", "response"]].apply(lambda x: f"""### Problem:
{x[0]}

### Solution:
{x[1]}""", axis=1)
print("****************************************")
print("Example prompt: \n", df["text"].iloc[0])
print("****************************************")
df_train, df_test = train_test_split(df, test_size=0.15)
trainer = finetune_trainer(
    model_name="stabilityai/stablelm-2-1_6b",
    x_train=df_train["text"].to_list(),
    x_test=df_test["text"].to_list(),
    training_args=dict(
        output_dir="math-finetune-model-stable-lm",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        num_train_epochs=1,
        learning_rate=1e-5,
        fp16=True,
        push_to_hub=True,
        # hub_token="pass_your_own_token"
    ),
    model_output_dir="my_pretrained_model"
)

trainer.train()
try:
    trainer.push_to_hub()
except:
    trainer.save()
