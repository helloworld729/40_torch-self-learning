# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2021/1/10 21:30
import traceback
from bs4 import BeautifulSoup

def parse(file_path, write_name):
    text = open("DATA/" + file_path, encoding="UTF-8")
    soup = BeautifulSoup(text, features="html.parser").find_all("doc")

    former = open("./" + write_name + ".former.txt", mode="w", encoding="UTF-8")
    latter = open("./" + write_name + ".latter.txt", mode="w", encoding="UTF-8")

    for i in range(len(soup)):
        try:
            summary = soup[i].summary.text
            short_text = soup[i].short_text.text
            former.write(short_text.replace("<br/>", "").strip() + "\n")
            latter.write(summary.replace("<br/>", "").strip() + "\n")
        # except Exception as e:
        except:
            print(soup[i])
            print("\n\n")

    text.close()
    former.close()
    latter.close()

print("INFO: begin")
print("INFO: train")
file_path = "PART_I.txt"
parse(file_path, "train")
print("INFO: valid")
file_path = "PART_II.txt"
parse(file_path, "valid")
print("INFO: test")
file_path = "PART_III.txt"
parse(file_path, "test")


