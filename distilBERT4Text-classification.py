#!/usr/bin/env python
# coding: utf-8

# # the 2nd part of the Work Flow to validate the product code provided by the importers
# # (through the standard product descriptions that corresponding to the PRDCT_ID) vs  
# # the importer's product description, using the BERT/TensorFlow/Huggingface in NLP
# 
# #the 1st part create the model, the 2nd part implement the bert model
# 
# Guozhen  Liu@Feb 24, 2021
# 
# 1) read into the Pandas the PD table which contains the FDA 
#    product description + manual curated product descriprions.
# 2) Read into Pandas the file contains the label to category number generated 
#    during model generation 
# 3) Load the tensorflow framework;
# 4) Load the BERT model and the BERT Tokenizer 
# 5) Read input file (CSV format) line by line, clean and correct some spelling errors
#    in the Importer's product description
# 6) get the top 5 match PRDCT_ID for each product description, assess the position of the 
#    importer provided PRDCT_ID
# 7) Append the results to the end of each line of the yearly transaction records     

# In[1]:


get_ipython().run_line_magic('reset', '-f')
import gc
gc.collect()

import pandas as pd
import re
import string
import csv

from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import json

import sys


# In[2]:


imptr_prod_desc = sys.argv[1]
output_file =  sys.argv[2]
top5_match = sys.argv[3]

fda_prod_desc = "C:/DataOcean/PRODUCT_DIMENSION_GZLmanual21821.xlsx"
save_model = "C:/DataOcean/output/saved_models/SeafoodTextClassifierByDistilBertFeb19"
cat2prd_label = "C:/DataOcean/output/SeafoodText4DistilBERTFeb19.csv" 


# In[3]:


def clean_text(s):
#Abbreviations:
       s = re.sub("\\bCLW\s*FRZ LOB\\b",    "CLAW FROZEN LOBSTER", s)
       s = re.sub("\\bFRZN?\\b",   "FROZEN", s)
       s = re.sub("\s+W\/G,",   " WILD GROWTH", s)
         
#Spelling errors or typos:    
       s = re.sub("\\bAMCHOV(Y|IES)\\b",    "ANCHOVY", s)
       s = re.sub("\\bAMCHIO?V(Y|ES|IES)\\b",    "ANCHOVY", s)
       s = re.sub("\\bANCHO(A|AS|NY|VE|BIES|DINA|IES|VES|VEY|VIE|VIES|VOES|VYS)\\b",    "ANCHOVY", s)  # European anchovy
       s = re.sub("\\bANCHHOVY\\b",    "ANCHOVY", s)
       s = re.sub("\\bANOCH(O|I)VY\\b",    "ANCHOVY", s)
       s = re.sub("\\bANVHO(CIES|VY)\\b",    "ANCHOVY", s)
       s = re.sub("\\bAN?CHO(VIES|VY|IVES|NY|VES|IVE)\\b",    "ANCHOVY", s)
       s = re.sub("\\bBOQUERO(N|NE|NES)\\b",    "ANCHOVY", s)
       s = re.sub("\\bCRAW?FISH\\b",  "CRAYFISH", s)
       s = re.sub("\\bGROUPPER",      "GROUPER", s)
       s = re.sub("\\bKINGCLIP\\b",   "KINGKLIP", s)          
       s = re.sub("\\bLOBESTER\\b",   "LOBSTER", s)
       s = re.sub("\\bMARKEREL\\b",   "MACKEREL", s)
       s = re.sub("\\bMACKREL\\b",    "MACKEREL", s)
       s = re.sub("\\bSHEEL\\b",      "SHELL", s)    
       s = re.sub("\\bSHIRMP\\b",     "SHRIMP", s)
       s = re.sub("\\bSRIMP\\b",      "SHRIMP", s)
       s = re.sub("\\bTILIPIA\\b",    "TILAPIA", s)
       s = re.sub("\\bFILET\\b",      "FILLET", s)    #Added Jan 8, 2021
       s = re.sub("\\bWHL\\b",      "WHOLE", s)    #Added Jan 8, 2021

#Digital numbers
       s = re.sub("\d+-?OYSTERS?\\b",    "OYSTERS", s)
       s = re.sub("\\bOYSTERS?-\d+",     "OYSTERS", s) 
       s = re.sub("\\bPKG\s+\d+X\d+(LB|OZ)\s+CANS\s*\/\s*\d+\s*MC\\b", "", s)  # PKG 12X1LB CANS / 1 MC
        
       s = re.sub("\d+\s+LB\.?\s+CAN\s+\d+\s+PK\\b",    "", s)
       s = re.sub("\d+-\d+\s*\w+\s+\w+\d+X\d+",   "", s)    
       s = re.sub("\d+\+\s*(OZ)?\\b", "", s)  # removing the "4+" anywhere  "4+ OZ" 
    
       s = re.sub("^\d+LB\/\S*\s+",   "", s)                           #10LB/1.6CM
       s = re.sub("\d+X\d+\s*LB\\b",    "", s)
       s = re.sub("\d+\s+-\s+\d+\s+GMS\/PCS\\b",   "", s)
       s = re.sub("\d+\w+-\w+\s+BAG\\b",   "", s)
       s = re.sub("\d+\s+LB\s+BOX\\b",   "", s)                     # 50 LB BOX
       s = re.sub("\d+-UP\s+OZ\\b",   "", s)

       s = re.sub("\s+U\d*(-|\/)\d+",   "", s)                         #U51/60    #U-15   U/15
       s = re.sub("\s+U\d+$",   "", s)                                 #U10  at end
  
       s = re.sub("\s*\d+(-|\/)\d+\s*(LB|OZ|KG|G|GM|GR)\\b",    "", s)
       s = re.sub("\(OG\)\d+\/(\d+|UN)\\b",    "", s)
       s = re.sub("\(\d+K?G\)",    "", s)                              #(100G)   (6KG)
       s = re.sub("\(\.\d+\s*LB\)",    "", s)                          #(.5 LB)
       s = re.sub("\\b\d+\s*LBS\\b", "", s)                              #30LBS
    
       s = re.sub("\d+\'S",    "", s)                                  # 25'S
       s = re.sub("\s+\d+\#$",    "", s)
       s = re.sub("\s+\#\d+$",    "", s)
       s = re.sub("\d+\s*OZ\\b",    "", s)
       s = re.sub("\(\d+(X|-)\d+\)",    "", s)          
       s = re.sub("\d+(-|\/|X)\d+",    "", s)

       s = re.sub("\d{2,}", "", s)     # removing the pure numbers anywhere
       s = re.sub("^\W+", "", s)          # removing the leading non-word 
       s = re.sub("\W+$", "", s)          # removing the trailing non-word 
       s = re.sub("\s\s+", " ", s)          # removing the reduntant spaces 
    
       return s


# In[4]:


fields =['PRDCT_ID', 'CLASS_CODE', 'GROUP_CODE', 'Desc', 'Prod_Label']
fda_desc = pd.read_excel(fda_prod_desc, usecols=fields)
total_prod_IDs = len(fda_desc)


# In[8]:


prodDict = {}
for index, row in fda_desc.iterrows():
    if  int(row['PRDCT_ID']) in prodDict:
        pass
    else:    
        prodDict[int(row['PRDCT_ID'])] = row['CLASS_CODE'] + str(row['GROUP_CODE'])                         + "_" + str(row['PRDCT_ID'])


# In[5]:


df = pd.read_csv(cat2prd_label, engine ='python')

cat2label = {}
for index, row in df.iterrows():
    if  row['encoded_cat'] in cat2label:
        pass
    else:    
        cat2label[row['encoded_cat']] = row['label']


# In[6]:


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
save_directory = save_model

#Loading the model and the tokenizer
loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)


# In[9]:


imptr_prod_desc = sys.argv[1]
output_file =  sys.argv[2]
top5_match = sys.argv[3]

#imptr_prod_desc = "C:/Users/Guozhen.Liu/Documents/Conceptant/PREDICT_External/LINES_2018_5000lines.txt"


outfile = open(output_file, "w")
fo = csv.writer(outfile, lineterminator='\n')
outfile2 = open(top5_match, "w")
fo2 = csv.writer(outfile2, lineterminator='\n')  #delimiter = '\t'
fo2.writerow(["Description","prodLabel","Rank inTop5"])

with open(imptr_prod_desc, newline='') as infile:
    reader = csv.reader(infile, delimiter=',', quotechar='"')
    headers = next(reader, None)  # returns the headers or `None` if the input is empty
    headers.append("Top5 Match")
    headers.append("Best Match Class/Group")
    headers.append("Best Match PRDCT_ID")
    headers.append("PRDCT_ID Rank in Top5")
    
    fo.writerow(headers)
    
    for words in reader:      
        PRDCT_ID = 0
        PIC_ID =  0
        try:
            PRDCT_ID = int(words[7].strip())    
   #         PIC_ID = int(words[14].strip()) 
        except (IndexError, ValueError):
            continue
    
        if ( PRDCT_ID in prodDict):   #Stem  and  PIC_ID in picDictStem) :  
            isInTop5 = 0
            test_text = clean_text(words[27].upper())
            predict_input = loaded_tokenizer.encode(test_text,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

            output = loaded_model(predict_input)[0]
        #    prediction_value = tf.argmax(output, axis=1).numpy()[0]  
        
            prediction_top5 = tf.math.top_k(output, k=5, sorted=True, name=None)[1].numpy()[0]
            prediction_top5score = tf.math.top_k(output, k=5, sorted=True, name=None)[0].numpy()[0]
 
            (prdctID1, prdctID2, prdctID3, prdctID4, prdctID5) = prediction_top5
            prdctID1 = cat2label[prdctID1]
            prdctID2 = cat2label[prdctID2]
            prdctID3 = cat2label[prdctID3]
            prdctID4 = cat2label[prdctID4]
            prdctID5 = cat2label[prdctID5]
            
            words.append(prediction_top5score)
            prodLabel = prodDict[PRDCT_ID]
            (prodCG, s1) = prodDict[prdctID1].split('_')  
            words.append(prodCG)
            words.append(prdctID1)
            
            if int(prdctID1) == PRDCT_ID:
                isInTop5 = 1
            elif int(prdctID2) == PRDCT_ID:
                isInTop5 = 2
            elif int(prdctID3) == PRDCT_ID:
                isInTop5 = 3
            elif int(prdctID4) == PRDCT_ID:
                isInTop5 = 4
            elif int(prdctID5) == PRDCT_ID:
                isInTop5 = 5
 
            words.append(isInTop5)
 
            fo.writerow(words)

            if  isInTop5 > 0 :    #High quality matches
                fo2.writerow([words[27], prodLabel,isInTop5])  
        
    
infile.close()
outfile.close()
outfile2.close()
#print("There are totally % 6d lines have internal line break!" %(bad_line_cnt))


# In[10]:


print("Done this Analysis!")