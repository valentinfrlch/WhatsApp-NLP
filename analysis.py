import os, zipfile, re
import nltk
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

import matplotlib.pyplot as plt


path = "dataset"

def extract(path):
    files_in_directory = os.listdir(path)
    filtered_files = [file for file in files_in_directory if file.endswith(".zip")]
    for file in filtered_files:
        with zipfile.ZipFile(os.path.join(path, file)   , 'r') as zip_ref:
            zip_ref.extractall("dataset/tmp")


def get_handles(lines):
    handles = []
    for names in lines:
        if len(handles) < 2:
            handle = names.split("]")[1].split(":")[0].strip()
            if handle not in handles:
                handles.append(handle)
        else:
            break
    return handles


def parse_message(line):
    message = ""
    messageSplit = line.split(":")
    messageList = []
    # if len(messageSplit) > 4:
    #     for i in range(3, len(messageSplit)):
    #         messageList.append(re.sub('<[^>]+>', '', messageSplit[i]).strip())
    #     message = ": ".join(messageList)
    # else:
    try:
        message = re.sub('<[^>]+>', '', messageSplit[3].strip()).replace("<attached", "").strip()
    except IndexError:
        print(messageSplit)
    return message

def get_text_by_handle(lines, handle):
    textList = []
    for line in lines:
        if handle in line:
            if parse_message(line) != " ":
                textList.append(parse_message(line))
    text = ". ".join(textList)
    return text

# --------------------- STATS ------------------------------

def absolute_count(lines):
    handles = get_handles(lines)
    a, b = 0, 0
    for go in lines:
        if handles[0] in go:
            a += 1
        if handles[1] in go:
            b += 1
    counts = [[handles[0], a], handles[1], b]
    return counts


def relative_count(lines):
    handles = get_handles(lines)
    a, b = 0, 0 # count in characters
    for go in lines:
        if handles[0] in go:
            a += len(parse_message(go))
        if handles[1] in go:
            b += len(parse_message(go))
    total = a + b
    a_per = a / total * 100
    b_per = b / total * 100
    counts = [[handles[0], a, a_per], handles[1], b, b_per]
    return counts


def linguistics(text):
    stop_words = [" ", ".", ",", "  "]
    tokenized_word=word_tokenize(text)
    dataset=[]
    for w in tokenized_word:
        if w not in stop_words:
            dataset.append(w)

    fdist = FreqDist(dataset)
    fdist.plot(30,cumulative=False)
    plt.show()

    return fdist.most_common(10)

#print(get_handles(lines))
#extract(path)
chat_file = open("dataset/tmp/_chat.txt", "r")
lines = chat_file.readlines()
#linguistics(get_text_by_handle(lines, "name"))

print(relative_count(lines))
