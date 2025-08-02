# # from transformers import pipeline
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel

# # # classifier = pipeline("sentiment-analysis")
# # # res = classifier("I've been waiting for a HuggingFace course my whole life.")

# # # generator = pipeline("text-generation", model="distilgpt2")
# # # res2 = generator("In this course I will teach you how to use the", max_length=30, num_return_sequences=2, truncation=True)

# # # classifier = pipeline("zero-shot-classification")
# # # res3 = classifier(
# # #     "This is a test",
# # #     candidate_labels=["food", "sports", "travel"],
# # #     multi_label=True,
# # # )

# # # print(res, res2, res3)

# # model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# # seq = "Hello, my dog is cute"
# # res = tokenizer(seq)
# # print(res)
# # tokens = tokenizer.tokenize(seq)
# # print(tokens)
# # ids = tokenizer.convert_tokens_to_ids(tokens)
# # print(ids)
# # decoded_string = tokenizer.decode(ids)
# # print(decoded_string)





# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import torch.nn.functional as F

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# X_train = ["I've been waiting for a HuggingFace course my whole life.", "Python is great!"]

# res = classifier(X_train)
# print(res)

# batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
# print(batch)

# with torch.no_grad():
#     outputs = model(**batch)
#     print(outputs)
#     predictions = torch.softmax(outputs.logits, dim=1)
#     print(predictions)
#     labels = torch.argmax(predictions, dim=1)
#     print(labels)

# save_directory = "saved"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
# tok = AutoTokenizer.from_pretrained(save_directory)
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)
