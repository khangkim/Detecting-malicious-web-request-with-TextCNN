from urllib.parse import unquote
import random
import csv

def analysis(filepath, bool):
    with open (filepath, "r") as file:
        datas = file.readlines()

    special_char = r"+-*/.,:;'=()<>%&?@~\"#$_|"

    url_list = []
    for data in datas:
        data = unquote(data, encoding='latin-1')
        data = unquote(data, encoding='latin-1')
        data = data.lower()
        data = data.replace("\n","")
        data = data.replace("\r","")
        url = ""
        url_bool = []
        for i in data:
            if i not in special_char:
                url += i
            else: 
                url += " "
                url += i 
                url += " "
        url_bool.append(url)
        url_bool.append(bool)
        url_list.append(url_bool)

    return url_list

url_attack = analysis("attack_url.txt", 1)
url_normal = analysis("normal_url.txt", 0)
url_all = url_attack + url_normal
random.shuffle(url_all)
n = len(url_all) // 5
n = n*4
url_train = url_all[:n]
url_test = url_all[n:]

with open("url_train.csv", "w" , newline = "",encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(url_train)

with open("url_test.csv", "w" , newline = "",encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(url_test)
