from __future__ import division
from psychopy import *
from numpy import *
from random import *
import time
from itertools import *
import os

def repite(frame,duration,numberStates=2,start=1): ### Creating repetitive stimuli
    startP=numberStates*duration-start%(numberStates*duration)
    counter=(frame+startP)%(numberStates*duration)
    for i in range(numberStates):
        if counter>=i*duration and counter<(i+1)*duration:
            flag=i+1
    return flag

def getResponse(keys):
    allKeys=event.getKeys(timeStamped = True)
    if len(allKeys) > 0:
        allK = [allKeys[0][0]]
        t = allKeys[0][1]
        for thisKey in allK:
            for k in keys:
                if thisKey == k:
                    return([t, k])
                if thisKey in ["q", "escape"]:
                    core.quit()
                
def waitResponse(keys):
    thisResp=None
    while thisResp==None:
        allKeys=event.waitKeys()
        for thisKey in allKeys:
            for k in keys:
                if thisKey==k:
                    return(k)
            if thisKey in ['q', 'escape']:
                core.quit()

def openDataFile(subject):
    if not os.path.exists('data'):
        os.makedirs('data')
    timeAndDateStr = time.strftime("%d%b%Y_%H-%M", time.localtime()) 
    dataFile=open('data/' + subject + timeAndDateStr  + '.txt', 'w')
    return dataFile
    
def createList(dicts):
    return list(dict(izip(dicts, x)) for x in product(*dicts.itervalues()))
