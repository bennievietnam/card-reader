import pandas as pd
import numpy as np
import nltk
import sklearn
# import os
# print(os.getcwd())

with open('storage/countries.txt', 'r') as f:
    lines = f.readlines()
countries = [l.strip() for l in lines]
with open('storage/type_permission.txt', 'r') as f:
    lines = f.readlines()
type_permission = [l.strip() for l in lines]


def jaccard_similarity(predicted, gold):
    intersection = set(predicted).intersection(set(gold))
    union = set(predicted).union(set(gold))
    return len(intersection)/len(union)

def ac(predicted, key):
    # print(f"input: {predicted}")
    if key == 'nationality':
        ac_db = countries
    elif key == 'type_permission':
        ac_db = type_permission
    result = []
    for c in ac_db:
        result.append(jaccard_similarity(predicted, c))
    # print(f"output: {countries[result.index(max(result))]}")
    return ac_db[result.index(max(result))]





# ac_country("xx- l ベルギーチンンa bd -*")
