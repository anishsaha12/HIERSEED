import pandas as pd
import xml.etree.ElementTree as ET

def parse_text(text):
    root = ET.XML(text)
    txt = root.find('.//text')
    res = []
    for child in txt:
        res.append(child.text)
    res = ' '.join(res)
    return res

df = pd.read_csv('rcv1_v2.csv')

df['parsed_text'] = df.text.apply(parse_text)
df.to_csv('rcv1_v2.csv',index=False)