from datasets import load_dataset

# load the sst2 dataset
dataset = load_dataset("glue", "sst2")

# select the train split
train = dataset["train"]

print("Examining train set:")
print(train)
print(train.features)

print("Information about the dataset:")
print(train.info.description)
print("Homepage")
print(train.info.homepage)

print("Examining sentence")
print(type(train["sentence"]))
print(type(train["sentence"][0]))


print("Examining label")
print(type(train["label"]))
print(type(train["label"][0]))
# set takes all the unique values
print(set(train["label"]))

print("A few samples:")
for t in range(10):
    sent = train["sentence"][t]
    lab = train["label"][t]
    print(sent, "-",  lab)