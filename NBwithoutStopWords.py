#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:53:04 2017

@author: vinaya
"""

import os
import re
#==============================================================================
# if(len(sys.argv)<3):
#     print("please input command line arguments:\n ")
#     print("<training-set> <test-set>")
#     exit
# else:
#==============================================================================
import io
import math

class NaiveBayes:
    def __init__(self,trainingHam,trainingSpam,testHam,testSpam):
        self.trainingHam = trainingHam
        self.trainingSpam = trainingSpam
        self.HAM = 0
        self.SPAM = 1 
        self.totalWordList = []
        self.hamCountDict = {}
        self.totalHamWordCount = 0
        self.spamCountDict = {}
        self.totalSpamWordCount = 0
        self.prior = {}
        self.totalCondProb = {}
        
        self.testHam = testHam
        self.testSpam = testSpam
        
        


    def getWordCountList(self,flag):   
        filepath = self.trainingHam
        if(flag == self.SPAM):
            filepath = self.trainingSpam
            
        trainingFile = os.listdir(filepath)
        countDict = {}
        totalWordCount = 0
        for each in trainingFile:
            #print(each)
            f = io.open(filepath+"/"+each, 'r',encoding='iso-8859-1')
            lines = f.readlines()
            for line in lines:
                #considering alpha numerics only
                lettersOnly = (re.sub("[^a-zA-Z\s]", "", line)).lower().split()
                for word in lettersOnly:
                    if word in countDict:
                        countDict[word]+=1
                    else:
                        countDict[word]=1
                    totalWordCount+=1
        wordList = list(countDict.keys())
        #f1.write(str(wordList))
        #Laplace addition
        totalWordCount+=len(wordList)
        if(flag == self.HAM):
            self.hamCountDict = countDict
            self.totalHamWordCount = totalWordCount
        else:
            self.spamCountDict = countDict
            self.totalSpamWordCount = totalWordCount
    
    def calcluateCondProb(self):
        
        for each in self.totalWordList:
            if(each in self.hamCountDict):
                hamCount = self.hamCountDict[each]
            else:
                hamCount = 0
            #Laplace addition
            hamCount+=1
            if(each in self.spamCountDict):
                spamCount = self.spamCountDict[each]
            else:
                spamCount =0
            #Laplace addition
            spamCount+=1
            
            self.totalCondProb[each]=[(float)(hamCount/self.totalHamWordCount),(float)(spamCount/self.totalSpamWordCount)]
            #f1.write(each+"\t\t\t"+str(self.totalCondProb[each][0])+" "+str(self.totalCondProb[each][1])+"\n")
        
        
    def run(self):      
        #for ham data
        self.getWordCountList(self.HAM)
        #for Spam data
        self.getWordCountList(self.SPAM)
        #total unique words
        self.totalWordList =set(list(self.hamCountDict.keys())+list(self.spamCountDict.keys()))
       
        
        
    def train(self):
        totalHamFiles = len(os.listdir(self.trainingHam))
        totalSpamFiles = len(os.listdir(self.trainingSpam))
        totalTrainingFiles = totalHamFiles+ totalSpamFiles
       
        #calculating prior
        self.prior[self.HAM] = totalHamFiles/totalTrainingFiles
        self.prior[self.SPAM] = totalSpamFiles/totalTrainingFiles
        self.calcluateCondProb()
   
    def getClassification(self,path):
        score = {self.HAM:0,self.SPAM:0}
        count = 0
        testFile =  os.listdir(path)

        for each in testFile: 

            f = io.open(path+"/"+each, 'r',encoding='iso-8859-1')
            fileData = f.read()
           
            lettersOnly = (re.sub("[^a-zA-Z\s]", "", fileData)).lower()
    
            wordList = set(lettersOnly.split())
            for eachClass in score.keys():
                score[eachClass] = math.log(self.prior[eachClass],2)
                for word in wordList:
                    if(word in self.totalWordList):
                        score[eachClass]+=math.log((self.totalCondProb[word])[eachClass],2)
            if(score[self.HAM]>score[self.SPAM]):
                if(path == self.testHam):
                    count+=1
                #f1.write(each+" is HAM\n")
            else:
                if(path == self.testSpam):
                    count+=1
                #f1.write(each+" is SPAM\n")
        return count
    


    def test(self):       

        hamTestResult = self.getClassification(self.testHam)
        hamTestFiles =  os.listdir(self.testHam)
        accuracy = (float)(hamTestResult/len(hamTestFiles))*100
        print("Ham test accuracy="+str(accuracy))

        spamTestFiles =  os.listdir(self.testSpam)
        spamTestResult = self.getClassification(self.testSpam)
        accuracy = (float)(spamTestResult/len(spamTestFiles))*100
        print("Spam test accuracy="+str(accuracy))
 
        totalAccuracy = (float)((hamTestResult+spamTestResult)/(len(spamTestFiles)+len(hamTestFiles)))*100
        print("Total test accuracy="+str(totalAccuracy))
        #f1.close()


    
#==============================================================================
# #f1 = open("debug.txt",'w')   
# nb = NaiveBayes("train/ham","train/spam","test/ham","test/spam")
# nb.run()
# nb.train()
# nb.test()
#==============================================================================
