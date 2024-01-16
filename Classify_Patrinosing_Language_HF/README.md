# NLP Project: Detect Patronising or Condescending language using Transformer-based Models from HuggingFace

Categories:
- Unbalanced power relations
- Shallow solution
- Presupposition
- Authority voice
- Metaphor
- Compassion
- The poorer, the merrier

Data format:
- Task 1: paragraph_id keyword country_code paragraph label
- Task 2: paragraph_id, paragraph, keyword, country_code, span_start, span_end, span_text, category_label, number_of_annotators_agreeing_in_that_label

Features: Country, Keyword

Useful links:
- **CW spec:** https://static.us.edusercontent.com/files/mCjbUPfgDjtNLLx4hdvs2TcF

- **Report Template:** https://www.overleaf.com/read/xhxbhtgjgbxv

- **Data paper:** https://aclanthology.org/2020.coling-main.518.pdf

- **Competition:** https://competitions.codalab.org/competitions/34344

- **Competition github:** https://github.com/Perez-AlmendrosC/dontpatronizeme/


Model improvements:
- cross validation, precision, recall, f1
- different pre-trained models (save trained models for ensembling)
- categorical injection at embedding
- hyperparameters: lr, lr_schedule, epochs, dropout(final layer)
- ensemble (average or try linear model)
- predict on original label
