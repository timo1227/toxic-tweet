import pandas as pd
import streamlit as st
import re
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel
from tqdm.auto import tqdm
import torch

MAX_LEN = 320
EPOCHS = 5
LEARNING_RATE = 1e-05
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DistilBertClass(torch.nn.Module):
    def __init__(self):
        super(DistilBertClass, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(torch.nn.Linear(768, 768),
                                              torch.nn.ReLU(),
                                              torch.nn.Dropout(0.1),
                                              torch.nn.Linear(768, 6))

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"youve", "you have", text)
    text = re.sub(r"donå«t", "do not", text)
    text = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    text = text.strip(' ')
    return text


class MultiLabelDataset(Dataset):

    def __init__(self, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.max_len = max_len

    def __len__(self):
        return 1  # Since we have only one input instance

    def __getitem__(self, index):
        text = str(self.text)

        inputs = self.tokenizer.encode_plus(text, None,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True)
        out = {
            "input_ids": torch.tensor(inputs['input_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(inputs['attention_mask'], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        }

        return out


def predict_tweet(tweet):
    tweet = clean_text(tweet)
    # Pass tweet to MultiLabelDataset
    dataset = MultiLabelDataset(tweet, tokenizer, MAX_LEN)
    data_loader = DataLoader(dataset, batch_size=1)

    predictions = []

    def prediction():
        model.eval()

        with torch.inference_mode():
            for _, data in tqdm(enumerate(data_loader, 0)):
                ids = data['input_ids'].to(DEVICE, dtype=torch.long)
                mask = data['attention_mask'].to(DEVICE, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(
                    DEVICE, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)
                probas = torch.sigmoid(outputs)

                predictions.append(probas)
        return probas

    prob = prediction()

    predictions = torch.cat(predictions)

    return predictions


model = DistilBertClass()
model.to(DEVICE)
model.load_state_dict(torch.load("model.pt"))

tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased', truncation=True, do_lower_case=True)


# STEAMLIT APP

def app():
    # create a title for the app
    st.title("Toxic Tweet Classification")

    # Drop down menu to select the model
    model = st.selectbox("Select the model", ["Finetuned DistilBert"])

    # create a text input for the user to enter their text
    tweet = st.text_input("Enter your text:",
                          value="I think this guy is a jerk")

    # create a button to classify the text
    if st.button("Classify"):
        # call the classifier pipeline to predict the sentiment of the text
        if model == "Finetuned DistilBert":
            results = predict_tweet(tweet)
        else:
            st.write("Please select a model")
        label_columns = ["Toxic", "Severe_toxic",
                         "Obscene", "Threat", "Insult", "Identity_hate"]
        st.header("Prediction Results")
        # Find the labels with the highest probability
        highest_prob = torch.max(results, dim=1).indices
        results = results.cpu().detach().numpy()
        # Disiplay Tweet | Label[highest_prob] | Probability in table
        st.table(pd.DataFrame(
            [[tweet, label_columns[highest_prob], results[0][highest_prob]]], columns=["Tweet", "Label", "Probability"]))


app()
