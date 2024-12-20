import sys
import numpy as np
import pickle
import pandas as pd
from fewshotWN import TTCA,TTfbCCA,TTstCCA
from utils import ITR
from tqdm import tqdm
import math
import os

respre=open('../dataset/pre.pickle','rb')
eegpre=pickle.load(respre)
resformal=open('../dataset/formal.pickle','rb')
eegformal=pickle.load(resformal)

subject = ['huchunjiang','lihaokun','lijiayang','liuyanhong','mengfanjie','miaoyining','ranlinglin','shinanlin','suyurou','xiemengshuo','xingyingying','yangyuxing','yefan','zhaoyadong','zhoumengying']

start_phase=np.tile(np.arange(0, 2, 0.5)*math.pi, 10)
frequency=np.linspace(8.0, 15.8, 40)
S_ssvep = []
t=np.arange(750)
for i in range(40):
    sti = np.sin( 2 * math.pi * frequency[i] /250 * t + start_phase[i] )
    S_ssvep.append(sti)
S_ssvep = np.stack(S_ssvep)


for subindex in tqdm(range(len(eegformal))):
    if not eegformal[subindex]['name'] in subject:
        continue
    
    frames=[]

    X_st_wn=[]
    y_st_wn=[]
    X_st_ssveps=[]
    y_st_ssveps=[]
    X_st_singles=[]
    y_st_singles=[]
    X_st_wns=[]
    y_st_wns=[]
    
    for subINX in range(len(eegformal)):
        if not eegformal[subINX]['name'] in subject:
            continue
        if eegformal[subINX]['name']==eegformal[subindex]['name']:
            continue
            
        chnNames = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'PO3', 'O2','P1','P2','P3','P4','P5','P6','P7','P8','PO7','PO8','CB1','CB2']
        chnINX = [eegformal[subINX]['channel'].index(i) for i in chnNames]
        X_single = eegpre[subINX]['stimulus']['X'][:,chnINX,:]
        y_single = eegpre[subINX]['stimulus']['y']-1
        X_wn = eegformal[subINX]['wn']['X'][:,chnINX,:]
        y_wn = eegformal[subINX]['wn']['y']-1
        X_ssvep = eegformal[subINX]['ssvep']['X'][:,chnINX,:]
        y_ssvep = eegformal[subINX]['ssvep']['y']-41
        X_st_wn.append(X_wn)
        y_st_wn.append(y_wn)
        X_st_ssveps.append(X_ssvep)
        y_st_ssveps.append(y_ssvep)
        X_st_singles.append(X_single)
        y_st_singles.append(y_single)
        X_st_wns.append(X_wn)
        y_st_wns.append(y_wn)
        
    X_st_wn = np.concatenate(X_st_wn,axis=0)
    y_st_wn = np.concatenate(y_st_wn,axis=0)
    X_st_wns = np.stack(X_st_wns)
    y_st_wns = np.stack(y_st_wns)
    X_st_ssveps = np.stack(X_st_ssveps)
    y_st_ssveps = np.stack(y_st_ssveps)
    X_st_singles = np.stack(X_st_singles)
    y_st_singles = np.stack(y_st_singles)
    
    chnNames = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'PO3', 'O2','P1','P2','P3','P4','P5','P6','P7','P8','PO7','PO8','CB1','CB2']
    chnINX = [eegformal[subindex]['channel'].index(i) for i in chnNames]

    X_single = eegpre[subindex]['stimulus']['X'][:,chnINX,:]
    y_single = eegpre[subindex]['stimulus']['y']-1
    S_single = eegpre[subindex]['stimulus']['STI']

    X_ssvep = eegformal[subindex]['ssvep']['X'][:,chnINX,:]
    y_ssvep = eegformal[subindex]['ssvep']['y']-41
    # S_ssvep = eegformal[subindex]['wn']['STI']
    
    X_wn = eegformal[subindex]['wn']['X'][:,chnINX,:]
    y_wn = eegformal[subindex]['wn']['y']-1
    S_wn = eegformal[subindex]['wn']['STI']
    
    winLENs=np.arange(0.25,3.25,0.25)
    
    for winLEN in winLENs:
        
        # model = TTCA(S_train = S_wn, S_test = S_wn, winLEN = winLEN, tmax=0.16, n_band=2, bandmethod=3, jitter=1,filterlearn=False)
        # model.fit(X_st_wn,y_st_wn)
        # score_UItemp = model.score(X_wn,y_wn)
        # itr_UItemp=ITR(40,score_UItemp,winLEN)
        
        
        model = TTstCCA(S_train = S_single, S_test = S_wn, stX_single=X_st_singles, sty_single=y_st_singles, stX_speller=X_st_wns, sty_speller=y_st_wns, winLEN = winLEN, tmax=0.16)
        model.fit(X_single,y_single)
        score_wn = model.score(X_wn,y_wn)
        itr_wn=ITR(40,score_wn,winLEN)
        
        model = TTstCCA(S_train = S_single, S_test = S_ssvep, stX_single=X_st_singles, sty_single=y_st_singles, stX_speller=X_st_ssveps, sty_speller=y_st_ssveps, winLEN = winLEN, tmax=0.16)
        model.fit(X_single,y_single)
        score_ssvep = model.score(X_ssvep,y_ssvep)
        itr_ssvep=ITR(40,score_ssvep,winLEN)
        
        
        # frame = pd.DataFrame({
        #                 'subject':[eegformal[subindex]['name']],
        #                 'winLEN':[winLEN],
        #                 'tag':['wn'],
        #                 'method':['UItemp'],
        #                 # 'algorithm':['TTCA'],
        #                 'score': [score_UItemp],
        #                 'ITR':[itr_UItemp]
        #             })
        # frames.append(frame)
        frame = pd.DataFrame({
                        'subject':[eegformal[subindex]['name']],
                        'winLEN':[winLEN],
                        'tag':['wn'],
                        'method':['UDtempUIdata'],
                        # 'algorithm':['FBCCA'],
                        'score': [score_wn],
                        'ITR':[itr_wn]
                    })
        frames.append(frame)
        frame = pd.DataFrame({
                        'subject':[eegformal[subindex]['name']],
                        'winLEN':[winLEN],
                        'tag':['ssvep'],
                        'method':['UDtempUIdata'],
                        # 'algorithm':['FBCCA'],
                        'score': [score_ssvep],
                        'ITR':[itr_ssvep]
                    })
        frames.append(frame)
        
        df = pd.concat(frames,axis=0,ignore_index=True)
        if not os.path.exists('../results/%s'%eegformal[subindex]['name']):
            os.makedirs('../results/%s'%eegformal[subindex]['name'])
        df.to_csv('../results/%s/transfer.csv'%eegformal[subindex]['name'])