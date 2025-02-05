import torch 
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from model_classes.first_party_model import FirstParty
from model_classes.third_party_model import ThirdParty
from model_classes.data_retention_model import DataRetention
from model_classes.policy_change_class import PolicyChange
from model_classes.user_access_class import UserAccess
from model_classes.user_choice_class import UserChoice

filepath = None

# load models 
rest_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # this tokenizer applies to a bunch of different models
config = DistilBertConfig.from_pretrained('distilbert-base-uncased') # config used for multiple models


first_party = FirstParty(num_labels_task1=4, num_labels_task2=16, num_labels_task3=11)
first_party.load_state_dict(torch.load('/mnt/data/models/first_party/first_party_model_state_dict.pth', map_location="cpu"))

third_party = ThirdParty(config, num_labels_task1=6, num_labels_task2=15, num_labels_task3=11)
third_party.load_state_dict(torch.load('/mnt/data/models/third_party/third_party_model.pth', map_location="cpu"))

data_retention = DataRetention(num_labels_task1=5, num_labels_task2=16, num_labels_task3=8)
data_retention.load_state_dict(torch.load('/mnt/data/models/data_retention/data_retention_model_state_dict.pth', map_location="cpu"))

policy_change = PolicyChange(config, num_labels_task1=5, num_labels_task2=5, num_labels_task3=6)
policy_change.load_state_dict(torch.load('/mnt/data/models/policy_change/policy_change_model_state_dict.pth', map_location="cpu"))

user_choice = UserChoice(config, num_labels_task1=9, num_labels_task2=5)
user_choice.load_state_dict(torch.load('/mnt/data/models/user_choice/user_choice_model_state_dict.pth', map_location="cpu"))

user_access = UserAccess(config, num_labels_task1=7, num_labels_task2=6)
user_access.load_state_dict(torch.load('/mnt/data/models/user_access/user_access_model_state_dict.pth', map_location="cpu"))


# load rest of the models same way

first_model = DistilBertForSequenceClassification.from_pretrained('/mnt/data/models/first_model_80')
first_model_tokenizer = DistilBertTokenizer.from_pretrained('/mnt/data/models/first_model_80')

data_security = DistilBertForSequenceClassification.from_pretrained('/mnt/data/models/data_security_model')
data_security_tokenizer = DistilBertTokenizer.from_pretrained('/mnt/data/models/data_security_model')

other_model = DistilBertForSequenceClassification.from_pretrained('/mnt/data/models/other_model')
other_model_tokenizer = DistilBertTokenizer.from_pretrained('/mnt/data/models/other_model')

inter_audiences = DistilBertForSequenceClassification.from_pretrained('/mnt/data/models/inter_audiences_model')
inter_audiences_tokenizer = DistilBertTokenizer.from_pretrained('/mnt/data/models/inter_audiences_model')

print("models loaded.")

# dict for 1-hot encodings

labels = {first_party: 0,
            third_party: 1,
            user_access: 2,
            data_retention: 3,
            data_security: 4,
            'International and Specific Audiences': 5,
            'Do Not Track': 6,
            'Policy Change': 7,
            'User Choice/Control': 8,
            'Introductory/Generic': 9,
            'Practice not covered': 10,
            'Privacy contact information': 11}


def classify_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for efficiency
        outputs = model(**inputs)
    
    logits = outputs.logits
    if model == first_model:
        probs = torch.sigmoid(logits)
        predicted_classes = (probs > 0.5).astype(int)
    else:    
        predicted_classes = (logits > 0.3).int()  # Get predicted label

    return predicted_classes



# Read the HTML file as text and split it into a list
with open(filepath, "r", encoding="utf-8") as file:
    content = file.read()

# Split the content based on '|||'
elements = content.split("|||")


for sent in elements:
    classification = classify_text(first_model, first_model_tokenizer, sent)
    print(classification)
    for i, clas in enumerate(classification):
        if clas == 1:
            print(i)

    break 

# WRITE FIRST AND THEN GO BACK AND FINISH THIS
# ADD AN EXAMPLE POLICY TO TEST THE MODEL 