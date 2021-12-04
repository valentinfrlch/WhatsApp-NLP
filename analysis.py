import os, zipfile, re
import nltk
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
import json
import matplotlib.pyplot as plt


path = "dataset"

def extract(path):
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall("dataset/tmp/")
    os.rename(path, path + "q")


def get_handles(lines):
    handles = []
    for names in lines:
        if len(handles) < 2:
            try:
                handle = names.split("]")[1].split(":")[0].strip()
            except IndexError:
                continue
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
        well = 1
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
    stop_words = [" ", ".", ",", "..", "\u200e"]
    tokenized_word=word_tokenize(text)
    dataset=[]
    for w in tokenized_word:
        if w not in stop_words:
            dataset.append(w)

    fdist = FreqDist(dataset)
    calculated_length = round(len(list(dict.fromkeys(dataset)))*0.25) # top 25%
    return fdist.most_common(calculated_length)

def linguistic_similarity(lines, handle1, handle2):
    text1 = get_text_by_handle(lines, handle1)
    text2 = get_text_by_handle(lines, handle2)
    dataset1 = linguistics(text1)
    dataset2 = linguistics(text2)
    set1, set2 = [], []
    for data in dataset1:
        set1.append(data[0])
    for data in dataset2:
        set2.append(data[0])
     
    a_vals = Counter(set1)
    b_vals = Counter(set2)

    words  = list(a_vals.keys() | b_vals.keys())
    a_vect = [a_vals.get(word, 0) for word in words]
    b_vect = [b_vals.get(word, 0) for word in words] 

    len_a  = sum(av*av for av in a_vect) ** 0.5
    len_b  = sum(bv*bv for bv in b_vect) ** 0.5
    dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))
    cosine = dot / (len_a * len_b)
    return cosine 


def analyze():
    files_in_directory = os.listdir(path)
    filtered_files = [file for file in files_in_directory if file.endswith(".zip")]

    check_stats = open("dataset/stats.txt", "r")
    existing_stats_lines = check_stats.readlines()
    existing_handles = []
    for l in existing_stats_lines:
        if "[" in l:
            h = l.strip().split(",")[0].replace("[", "")
            existing_handles.append(h)

    for file in filtered_files:
        print("[INFO] Analyzing " + file)
        extract(os.path.join("/home/pi/scripts/WhatsApp Analysis/dataset/", file))
        chat_file = open("/home/pi/scripts/WhatsApp Analysis/dataset/tmp/_chat.txt", "r")
        lines = chat_file.readlines()
        handles = get_handles(lines)
        stats = open("dataset/stats.txt", "a")

        if handles[0] not in existing_handles:
            stats.write("[" + handles[0] + ", " + handles[1] + "]\n")
            stats.write("linguistic_similarity: " + str(linguistic_similarity(lines, handles[0], handles[1])) + "\n")
            stats.write("relative_count: " + str(relative_count(lines)) + "\n")
            stats.write("absolute_count: " + str(absolute_count(lines)) + "\n")
            stats.close()

        # clean up
        for f in os.listdir("dataset/tmp/"):
            file_path = os.path.join("dataset/tmp", f)
            os.remove(file_path)

analyze()
