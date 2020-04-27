import requests
import csv
from bs4 import BeautifulSoup
import os
import time
import datetime
import re
import pandas as pd

now = datetime.date.today()
flag = False

### Initialized
with open(os.path.join("data","train.csv"),"wt",encoding='utf_8_sig') as f :
    writer = csv.writer(f)
    writer.writerow(['w_judge','e_judge','w_rank','w_name','ruler','e_rank','e_name','year','month','day'])
with open(os.path.join("data","test.csv"),"wt",encoding='utf_8_sig')  as f :
    writer = csv.writer(f)
    writer.writerow(['w_judge','e_judge','w_rank','w_name','ruler','e_rank','e_name','year','month','day'])

def makecsv (year,month) :
    for day in range(1,16):
        while True :
            print(str(year) + "," + str(month) + "," + "{:02}".format(day))
            # url = "http://sumodb.sumogames.de/Results.aspx?b=" + str(year) + str(month) + "&d=" + str(day)
            url = "http://sumodb.sumogames.de/Results.aspx?b=" + str(year) + str(month) + "&d=" + str(day) + "&l=j"
            html = requests.get(url)
            # print(html.status_code)
            if html.status_code == 404 :
                continue
            html = BeautifulSoup(html.text, 'html.parser')
            if str(year) != str(now.year) :
                csvFile = open(os.path.join("data","train.csv"), 'at', encoding='utf_8_sig')
            else :
                csvFile = open(os.path.join("data","test.csv"), 'at', encoding='utf_8_sig')
            writer = csv.writer(csvFile)
            soup = html.find("td",{"class":"layoutright"})
            # soup = html.find("div",{"class":"mb20p","id":"makuuchi"})
            if soup is None:
                continue
            soup = soup.findAll("table")
            for table in soup:
                for rows in table.findAll(['tr']):
                    csvRow = []
                    for cell in rows.findAll('td',{'class':'tk_kaku'}) :
                        if re.search('Makuuchi',str(cell.get_text())) or re.search('Juryo',str(cell.get_text())) or re.search('幕内',str(cell.get_text())) or re.search('十両',str(cell.get_text())):
                            flag = True
                        elif re.search('Makushita',str(cell.get_text())) or re.search('Sandanme',str(cell.get_text())) or re.search('Jonidan',str(cell.get_text())) or re.search('Jonokuchi',str(cell.get_text())) or re.search('幕下',str(cell.get_text())) or re.search('三段目',str(cell.get_text())) or re.search('序二段',str(cell.get_text())) or re.search('序の口',str(cell.get_text())) :
                            flag = False
                    if flag is True :
                        for cell in rows.findAll('td',{"class":"tk_kekka"}):
                            if re.search('kuro',str(cell)) is not None :
                                csvRow.append('0')
                            elif re.search('shiro',str(cell)) is not None :
                                csvRow.append('1')
                            elif re.search('fusenpai',str(cell)) is not None :
                                csvRow.append('NaN')
                            elif re.search('fusensho',str(cell)) is not None :
                                csvRow.append('NaN')
                        for cell in rows.findAll('td',{"class":"tk_east"}):
                            for span in cell.findAll('font',{"size":"1"}) :
                                if re.search('color',str(span)) is None :
                                    csvRow.append(span.get_text())
                            for span in cell.findAll('a') :
                                if re.search('color',str(span)) is None :
                                    csvRow.append(span.get_text())
                        for cell in rows.findAll('td',{"class":"tk_kim"}):
                            cell = re.sub(r'<td class="tk_kim">.*</a><br/>',r'',str(cell))
                            cell = re.sub(r'<td class="tk_kim">.*<br/></font>',r'',str(cell))
                            cell = re.sub(r'<br/><font color="#000000" size="1">.*</td>',r'',str(cell))
                            csvRow.append(cell)
                        for cell in rows.findAll('td',{"class":"tk_west"}):
                            for span in cell.findAll('font',{"size":"1"}) :
                                if re.search('color',str(span)) is None :
                                    csvRow.append(span.get_text())
                            for span in cell.findAll('a') :
                                if re.search('color',str(span)) is None :
                                    csvRow.append(span.get_text())
                    if csvRow != [] :
                        csvRow.extend([year,month,day])
                        writer.writerow(csvRow)
                    # print(csvRow)
            csvFile.close()
            time.sleep(1)
            break

### ~ last year
for year in range(2001,now.year):
    basyo = ["01","03","05","07","09","11"]
    for month in basyo :
        makecsv(year,month)

### This year
if now.month >= 1 :
    makecsv(str(now.year),"01")
