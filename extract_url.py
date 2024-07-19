import re
import string

def extract(data_path, file_save):
    with open(data_path,"r") as file:
        lines = file.readlines()

    urls = []
    post = []
    a = False

    for line in lines:
        if line.startswith("GET"):
            url = line.split(" ")
            urls.append(url[1].strip(' \n'))

        if line.startswith("POST"):
            a = True
        if a: 
            post.append(line)
        if a & line.startswith("End") :
            b = post[0].split(" ")
            url = b[1] + post[-3] 
            urls.append(url.strip(' \n'))
            post.clear()
            a = False

    new_urls = list(set(urls))
    with open(file_save,"w") as file:
        for url in new_urls:   
            file.write(url.lstrip('https://')+ "\n")

extract("./database/cisc_anomalousTraffic_test.txt", "./database/attack_url.txt")
extract("./database/cisc_normalTraffic_test.txt", "./database/normal_url.txt")
































