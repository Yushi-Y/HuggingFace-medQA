import os
import numpy as np

# helper function to save predictions to an output file
def labels2file(predictions, file_name):
    with open(file_name, "w") as f:
        f.write('\n'.join([str(x) for x in predictions]))

def majority_vote(folder_name):
    predictions = []
    for file in os.listdir('predictions'):
        path = f'{folder_name}/{file}'
        with open(path, 'r') as f:
            preds = f.read().split('\n')
            predictions.append(preds)
    predictions = np.array(predictions, dtype=int)
    predictions = (predictions.sum(axis=0) > len(predictions) / 2).astype(int)
    labels2file(predictions, 'task1.txt')


# from simpletransformers.classification import ClassificationArgs, ClassificationModel
# from sklearn.metrics import classification_report
# task1_model_args = ClassificationArgs(num_train_epochs=1, 
#                                       no_save=True, 
#                                       no_cache=True, 
#                                       overwrite_output_dir=True)
# task1_model = ClassificationModel("distilbert", 
#                                   'distilbert-base-uncased', 
#                                   args = task1_model_args, 
#                                   num_labels=2, 
#                                   use_cuda=True)
# task1_model.train_model(train_df[['text', 'target_label']])
# # run predictions
# preds_task1, _ = task1_model.predict(val_df['text'].tolist())
# print(classification_report(val_df['target_label'].to_numpy(), preds_task1))


# adafactor_beta1: float = None
# adafactor_clip_threshold: float = 1.0
# adafactor_decay_rate: float = -0.8
# adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
# adafactor_relative_step: bool = True
# adafactor_scale_parameter: bool = True
# adafactor_warmup_init: bool = True
# adam_epsilon: float = 1e-8

# config: dict = field(default_factory=dict)
# cosine_schedule_num_cycles: float = 0.5
# custom_layer_parameters: list = field(default_factory=list)
# custom_parameter_groups: list = field(default_factory=list)

# evaluate_during_training: bool = False
# evaluate_during_training_silent: bool = True
# evaluate_during_training_steps: int = 2000
# evaluate_during_training_verbose: bool = False
# evaluate_each_epoch: bool = True
# fp16: bool = True
# gradient_accumulation_steps: int = 1
# learning_rate: float = 4e-5
# local_rank: int = -1
# logging_steps: int = 50
# loss_type: str = None
# loss_args: dict = field(default_factory=dict)
# manual_seed: int = None

# polynomial_decay_schedule_lr_end: float = 1e-7
# polynomial_decay_schedule_power: float = 1.0
# process_count: int = field(default_factory=get_default_process_count)
# quantized_model: bool = False
# reprocess_input_data: bool = True


# scheduler: str = "linear_schedule_with_warmup"
# silent: bool = False
# skip_special_tokens: bool = True

# thread_count: int = None

# train_custom_parameters_only: bool = False
# use_cached_eval_features: bool = False

# use_hf_datasets: bool = False


# warmup_ratio: float = 0.06
# warmup_steps: int = 0
# weight_decay: float = 0.0