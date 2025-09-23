import pandas as pd
import csv

with open('read.txt', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)
data = pd.read_csv('read.txt')
print(data) 

data.to_csv('write.txt', index=False)
data.to_csv('write_no_header.txt', index=False, header=False)               

