import numpy as np
from scipy import stats, signal
from scipy.stats import zscore
from mne.decoding.receptive_field import _delay_time_series, _times_to_delays
from sklearn.metrics import accuracy_score
import math
from sklearn.cross_decomposition import CCA

class TTCA():
    def __init__(self, S_train, S_test, winLEN = 3.0, srate = 250.0, alpha = 0.9, tmin = 0.0, tmax = 0.9, n_band = 5, bandmethod=3, jitter=2,filterlearn=False,labels_test = np.arange(40)):
        self.winLEN = winLEN
        self.n_band = n_band
        self.srate = srate
        self.latency = round(0.14*self.srate)
        self.alpha = alpha
        self.tmin = tmin
        self.tmax = tmax
        self.fill_mean = True
        self.bandmethod=bandmethod
        S_train = (S_train-np.min(S_train,axis=-1,keepdims=True))/(np.max(S_train,axis=-1,keepdims=True)-np.min(S_train,axis=-1,keepdims=True))*2-1
        if S_test is not None:
            S_test = (S_test-np.min(S_test,axis=-1,keepdims=True))/(np.max(S_test,axis=-1,keepdims=True)-np.min(S_test,axis=-1,keepdims=True))*2-1
        self.S_train = S_train
        self.S_test = S_test
        self.jitter = jitter
        self.labels_test = labels_test

    
    def filterbank(self,x,freqInx):

        srate = self.srate/2
        
        if self.bandmethod==0:      #ssvep
            # passband1 = [2, 14, 22, 30, 38, 46, 54, 62, 70, 78]
            # stopband1 = [1, 10, 16, 24, 32, 40, 48, 56, 64, 72]
            # passband2 = [124,124,124,124,124,124,124,124,124,124]
            # stopband2 = [125,125,125,125,125,125,125,125,125,125]
            
            passband1 = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
            stopband1 = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
            passband2 = [90,90,90,90,90,90,90,90,90,90]
            stopband2 = [100,100,100,100,100,100,100,100,100,100]
            
            # passband1 = [6]
            # stopband1 = [4]
            # passband2 = [34]
            # stopband2 = [40]
            
        if self.bandmethod==1:
            passband1=[6,6,6,6,6]
            stopband1=[4,4,4,4,4]
            passband2=[18,26,34,42,50]
            stopband2=[24,32,40,48,56]
        
        elif self.bandmethod==2:
            passband1=[6,14,22,30,38]
            stopband1=[4,10,16,24,32]
            passband2=[18,26,34,42,50]
            stopband2=[24,32,40,48,56]
        
        elif self.bandmethod==3:
            passband1=[6,14,22,30,38]
            stopband1=[4,10,16,24,32]
            passband2=[18,34,50,50,50]
            stopband2=[24,40,56,56,56]
        
        elif self.bandmethod==4:
            passband1=[6,14,22,30,38]
            stopband1=[4,10,16,24,32]
            passband2=[50,50,50,50,50]
            stopband2=[56,56,56,56,56]
        
        
        Wp = [passband1[freqInx]/srate, passband2[freqInx]/srate]
        Ws = [stopband1[freqInx]/srate, stopband2[freqInx]/srate]
        
        [N, Wn]=signal.cheb1ord(Wp, Ws, 3, 40)
        [B, A] = signal.cheby1(N, 0.5, Wn,'bandpass')
        
        filtered_signal = np.zeros(np.shape(x))
        if len(np.shape(x))==2:
            for channelINX in range(np.shape(x)[0]):
                filtered_signal[channelINX,:] = signal.filtfilt(B, A, x[channelINX, :])
        elif len(np.shape(x))==3:
            for epochINX,epoch in enumerate(x):
                for channelINX in range(np.shape(epoch)[0]):
                    filtered_signal[epochINX,channelINX,:] = signal.filtfilt(B, A, epoch[channelINX, :])

        return filtered_signal
    
    def fit(self, X, y):
        winLENs = round(X.shape[-1]*self.srate)
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)
        S_train = self.S_train[:,self.latency:self.latency+winLENs]
        
        self.filters,_=self.getSpatialFilters(X,y)
        
        self.epochs = X
        self.labels = y
        
        self.TRF=self.getTRF(self.evokeds, S_train)     #fb*tau
        self.rctX = self.getrctX(self.S_test)       #fb*class*T

        return 
    
    def predict(self, X):
        
        winLENs = round(self.winLEN*self.srate)
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)

        rctX = self.rctX[...,self.latency:self.latency+winLENs]
        rctX = rctX.transpose(1,0,-1)      #class*fb*T
        rctXs = _delay_time_series(rctX.transpose(-1,0,1), -self.jitter/self.srate, self.jitter/self.srate, self.srate,fill_mean=True)      #T*class*fb*lag
        rctXs = rctXs.transpose(1,-1,2,0)   #class*lag*fb*T
        
        X_addfb = []
        for fbINX in range(self.n_band):
            X_addfb.append(self.filterbank(X, fbINX))
        X_addfb = np.stack(X_addfb)     # fb*epoch*channel*T
        X_addfb = np.transpose(X_addfb,(1,0,-2,-1))
        
        epochNUM, _, _ = np.shape(X)
        classNUM, lagNUM, _, _ = np.shape(rctXs)
        
        fb_coefs = np.expand_dims(np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)    #-1.25,0.25
        
        result = []
        corrmatrix = np.zeros((epochNUM, classNUM))
        
        for epochINX, epoch in enumerate(X_addfb):
            # rlagINX = np.zeros((classNUM,1)).astype('int')
            for classINX, template in enumerate(rctXs):
                rlag = np.zeros((lagNUM,1))
                for lagINX, laggedtemp in enumerate(template):
                    rtemp = np.zeros((self.n_band,1))
                    for fbINX, (fbepoch, fbtemplate, fbfilter) in enumerate(zip(epoch, laggedtemp, self.filters)):
                        rtemp[fbINX,:] = np.corrcoef(fbtemplate.reshape(1,-1), (fbfilter.T.dot(fbepoch)).reshape(1,-1))[0,1]
                    rlag[lagINX,:] = fb_coefs.dot(rtemp)
                r=np.max(rlag)
                # rlagINX[classINX,:]=np.argmax(rlag)
                corrmatrix[epochINX,classINX]=r
            result.append(self.labels_test[np.argmax(corrmatrix[epochINX,:])])
        self.result = np.stack(result)
        self.corrmatrix = corrmatrix
        
        return self.result
        
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    
    def getSpatialFilters(self,X,y):
        labels = np.unique(y)
        epochNUM, self.channelNUM, _ = np.shape(X)
        
        X_addfb = []
        for fbINX in range(self.n_band):
            X_addfb.append(self.filterbank(X, fbINX))
        X_addfb = np.stack(X_addfb)     #fb*epoch*channel*T
        
        X_classified = []
        for fbX in X_addfb:
            augmentClass = []
            for _, label in enumerate(labels):
                this_class_data =fbX[y == label]
                augmentEpoch = []
                for epoch in this_class_data:
                    augmentEpoch.append(epoch)
                augmentClass.append(np.stack(augmentEpoch))
            X_classified.append(augmentClass)       #fb*class*block*channel*T
        X_classified = np.stack(X_classified)

        augmentEvoked = []
        for fbs in X_classified:
            augmentEvoked.append([con.mean(axis=0) for con in fbs])
        augmentEvoked = np.stack(augmentEvoked)
        
        filters=[]
        for (fbEvoked, fbEpochs) in zip(augmentEvoked, X_classified):
            # norm
            fbEvoked = fbEvoked-np.mean(fbEvoked,axis=-1,keepdims=True)
            fbEvokedFeature = np.mean(fbEvoked, axis=0, keepdims=True)
            betwClass = fbEvoked-fbEvokedFeature
            betwClass = np.concatenate(betwClass,axis=1)    #channel*(block*T)
            # norm
            fbEpochs = [this_class-np.mean(this_class, axis=-1, keepdims=True) for this_class in fbEpochs]
            # allClassEvoked = [this_class-np.mean(this_class, axis=0, keepdims=True) for this_class in fbEpochs]
            allClassEvoked = fbEpochs
            allClassEvoked = [np.transpose(this_class,axes=(1,2,0)) for this_class in allClassEvoked]
            allClassEvoked = [np.reshape(this_class, (self.channelNUM, -1),order='F') for this_class in allClassEvoked]
            allClassEvoked = np.hstack(allClassEvoked)

            
            Hb = betwClass/math.sqrt(len(labels))
            Hw = allClassEvoked/math.sqrt(epochNUM)
            Sb = np.dot(Hb,Hb.T)
            Sw = np.dot(Hw, Hw.T)
            
            C = np.linalg.inv(Sw).dot(Sb)
            lamda, W = np.linalg.eig(C)

            idx=lamda.argsort()[::-1]
            W = W[:,idx]
            filter=np.squeeze(W[:,0:1])
            filters_corrected=filter*np.sign(filter[np.argmax(abs(filter),axis=0)])
            
            filters.append(filters_corrected)
        filters = np.stack(filters)       #fb*channel
        
        
        evokeds=[]
        for _, (fbfilter, fbevoked) in enumerate(zip(filters, augmentEvoked)):
            fbfilter = fbfilter[np.newaxis,:]
            evoked = []
            for _, epoch in enumerate(fbevoked):
                evoked.append(np.squeeze(fbfilter.dot(epoch)))
            evokeds.append(evoked)
        evokeds = np.stack(evokeds)    #fb*class*T
        self.evokeds = evokeds
        
        return filters, evokeds

            
    def getTRF(self,X,S):
        TRF_allepochs=[]
        TRF = []
        _, epochNUM, _, = np.shape(X)
        laggedLEN = len(_times_to_delays(self.tmin,self.tmax,self.srate))

        for fbINX, fbX in enumerate(X):
            fbKernel = np.zeros((epochNUM,laggedLEN))
            fbCov_sr = np.zeros((epochNUM,laggedLEN))
            
            fbS = self.filterbank(S,fbINX)

            for epochINX,(epoch,sti) in enumerate(zip(fbX,fbS)):
                sti = sti[:,np.newaxis]
                epoch = epoch[np.newaxis,:]
                laggedS = _delay_time_series(sti, self.tmin, self.tmax,self.srate,fill_mean=self.fill_mean).squeeze()
            
                # stimulation whitening
                Cov_ss = laggedS.T.dot(laggedS)
                u,sigma,v = np.linalg.svd(Cov_ss)
                for i in range(len(sigma)):
                    if sum(sigma[0:len(sigma)-i]/sum(sigma)) < self.alpha:
                        sigma = 1/sigma
                        sigma[len(sigma)-i:] = 0
                        break
                sigma_app = np.diag(sigma)
                inv_C = u.dot(sigma_app).dot(v)
                
                
                fbCov_sr[epochINX,:] = np.squeeze(epoch.dot(laggedS))
                fbKernel[epochINX,:] = np.squeeze(epoch.dot(laggedS).dot(inv_C.T))

            TRF_allepochs.append(fbKernel)
            TRF.append(np.mean(fbKernel,axis=0))
            
        self.TRF_allepochs=np.stack(TRF_allepochs)     #fb*class*tau
        TRF = np.stack(TRF)   #fb*tau
        
        return TRF
    

    def getrctX(self, S):

        rctX = []
        for fbINX,fbtrf in enumerate(self.TRF):
            fbrX = []
            fbS = self.filterbank(S,fbINX)
            fbtrf = fbtrf[np.newaxis,:]
            for _, sti in enumerate(fbS):
                sti = sti[:,np.newaxis]
                laggeds = np.squeeze(_delay_time_series(sti,tmin = self.tmin, tmax = self.tmax, sfreq = self.srate))
                fbrX.append(np.squeeze(fbtrf.dot(laggeds.T)))
            rctX.append(fbrX)
        rctX = np.stack(rctX)       #fb*class*T
        return rctX
    
    
    
    
class TTstCCA(TTCA):
    def __init__(self, S_train, S_test, stX_single, sty_single, stX_speller, sty_speller, winLEN = 3.0, srate=250, alpha=0.9, tmin=0, tmax=0.9):
        super().__init__(S_train=S_train, S_test = S_test, winLEN=winLEN, srate=srate, alpha=alpha, tmin=tmin, tmax=tmax)
        #w1对应fewshot，w2对应crosssub
        self.stX_single = stX_single
        self.sty_single = sty_single
        self.stX_speller = stX_speller
        self.sty_speller = sty_speller
        self.subNUM = np.shape(stX_single)[0]
        self.n_band=5
        self.bandmethod=0
        
    def fit(self,X,y):
        winLENs = round(X.shape[-1]*self.srate)
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)
        self.stX_single = self.stX_single[...,self.latency:self.latency+winLENs]
        self.stX_single = self.stX_single-np.mean(self.stX_single,axis=-1,keepdims=True)
        self.stX_speller = self.stX_speller[...,self.latency:self.latency+winLENs]
        self.stX_speller = self.stX_speller-np.mean(self.stX_speller,axis=-1,keepdims=True)

        S_test = self.S_test[:,self.latency:self.latency+winLENs]
        
        stfilters_single,stevokeds_single = self.stfit(self.stX_single,self.sty_single)
        stevokeds_speller = self.getstevokeds(stfilters_single,self.stX_speller,self.sty_speller)
        # _,stevokeds_speller = self.stfit(self.stX_speller,self.sty_speller)
        stevokeds_single = stevokeds_single.transpose(1,0,-2,-1)        #fb*sub*class*T
        stevokeds_speller = stevokeds_speller.transpose(1,0,-2,-1)        #fb*sub*class*T
        _,_,self.classNUM,_ = np.shape(stevokeds_speller)
        
        self.filters_st,self.evokeds_st=self.getSpatialFilters(X,y)
        
        stfilters=[]
        
        templates=[]
        for _,(fbstevoked,fbevoked,fbstevokeds) in enumerate(zip(stevokeds_single,self.evokeds_st,stevokeds_speller)):
            sttemplate = fbstevoked.reshape(self.subNUM,-1).T
            sttemplates = fbstevokeds.reshape(self.subNUM,-1).T
            template = fbevoked.reshape(-1,1)
            stfilter=np.linalg.inv(sttemplate.T.dot(sttemplate)).dot(sttemplate.T).dot(template)
            templates.append(sttemplates.dot(stfilter).reshape(self.classNUM,-1))
            stfilters.append(stfilter)
        templates = np.stack(templates)
        stfilters = np.stack(stfilters)
        self.stfilters = stfilters
        
        self.templates = templates
        
        self.TRF_crosssub = self.getTRF(self.templates, S_test)
        return
    
    
    def predict(self, X):
        winLENs = round(self.winLEN*self.srate)
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)
        templates = self.templates[...,:winLENs].transpose(1,0,-1)    #class*fb*T

        X_addfb_st = []
        for fbINX in range(self.n_band):
            X_addfb_st.append(self.filterbank(X, fbINX))
        X_addfb_st = np.stack(X_addfb_st)     # fb*epoch*channel*T
        X_addfb_st = np.transpose(X_addfb_st,(1,0,-2,-1))      #epoch*fb*channel*T
        

        epochNUM, _, _ = np.shape(X)
        classNUM, _, _ = np.shape(templates)
        
        fb_coefs_st = np.expand_dims(np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)
        
        result = []
        corrmatrix = np.zeros((epochNUM, classNUM))
        for epochINX, stepoch in enumerate(X_addfb_st):
            for classINX, sttemplate in enumerate(templates):
                rtemp = np.zeros((self.n_band,1))
                for fbINX,(fbstepoch,fbsttemplate,fbfilter) in enumerate(zip(stepoch,sttemplate,self.filters_st)):
                    rtemp[fbINX,:] = np.corrcoef(fbsttemplate,fbfilter.T.dot(fbstepoch))[0,1]
                r = fb_coefs_st.dot(rtemp)
                corrmatrix[epochINX,classINX]=r
            result.append(np.argmax(corrmatrix[epochINX,:]))
        self.result = np.stack(result)
        self.corrmatrix = corrmatrix
        
        return self.result
        
    def stfit(self,X,y):
        # X:sub*epoch*channel*T
        # y:sub*epoch
        stevokeds=[]
        stfilters=[]
        for _, (subX, suby) in enumerate(zip(X,y)):
            stfilter, stevoked = self.getSpatialFilters(subX,suby)
            #stevoked = fb*class*T
            #stfilters = fb*channel
            stevokeds.append(stevoked)
            stfilters.append(stfilter)
        stevokeds = np.stack(stevokeds)
        stfilters = np.stack(stfilters)
        return stfilters, stevokeds
    
    def getstevokeds(self,w,X,y):
        
        stfilters_test,stevokeds = self.stfit(X,y)
        stevokeds_resign = []
        for subINX, (substfilters_train, substfilters_test) in enumerate(zip(w,stfilters_test)):
            fbstevokeds=[]
            for fbINX,(fbtrain, fbtest) in enumerate(zip(substfilters_train,substfilters_test)):
                if np.linalg.norm(fbtrain-fbtest)>np.linalg.norm(fbtrain+fbtest):
                    fbstevokeds.append(-stevokeds[subINX,fbINX])
                else:
                    fbstevokeds.append(stevokeds[subINX,fbINX])
            stevokeds_resign.append(fbstevokeds)
        stevokeds_resign=np.stack(stevokeds_resign)
        return stevokeds_resign
        


class TTfbCCA(TTCA):
    def __init__(self, S_train, winLEN = 3.0, srate=250, alpha=0.9, tmin=0, tmax=0.9, n_band=5, bandmethod=0):
        super().__init__(S_train=S_train, S_test = None, winLEN=winLEN, srate=srate, alpha=alpha, tmin=tmin, tmax=tmax, n_band=n_band, bandmethod=bandmethod)
    
    
    def fit(self, X, y):
        winLENs = round(X.shape[-1]*self.srate)
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)
        S_train = self.S_train[:,self.latency:self.latency+winLENs]
        
        self.filters,self.evokeds = self.getSpatialFilters(X,y)
        self.TRF = self.getTRF(self.evokeds, S_train)     #fb*tau
        return 
    
    
    def gettemplate(self,trialLength=3.0):
        frequency=np.linspace(8.0, 15.8, 40)
        stim_ssvep_sincos = []
        t=np.arange(round(self.srate*trialLength))
        for freqINX in range(2*self.n_band):
            stim=[]
            for i in range(40):
                if freqINX % 2 ==0:
                    sti = np.sin( 2 * math.pi * frequency[i] * float(((freqINX+2)//2)) /250 * t  )
                else:
                    sti = np.cos( 2 * math.pi * frequency[i] * float(((freqINX+1)//2)) /250 * t  )
                stim.append(sti)
            stim_ssvep_sincos.append(stim)
        stim_ssvep_sincos = np.stack(stim_ssvep_sincos)
        rctXs=[]
        for freqINX in range(2*self.n_band):
            rctX=self.getrctX(stim_ssvep_sincos[freqINX,:,:])
            rctXs.append(rctX)
        rctXs = np.stack(rctXs)
        self.rctX = rctXs
        
        return self.rctX
    
    
    def predict(self, X):
        winLENs = round(self.winLEN*self.srate)
        trialLength = X.shape[-1]/self.srate
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)
        
        self.rctX = self.gettemplate(trialLength=trialLength)

        rctX = self.rctX[...,self.latency:self.latency+winLENs]    #fq*fb*class*T
        rctX = np.transpose(rctX,(2,1,0,-1))      #class*fb*fq*T

        epochNUM, _, _ = np.shape(X)
        classNUM, _, _, _ = np.shape(rctX)
        
        X_addfb = []
        for fbINX in range(self.n_band):
            X_addfb.append(self.filterbank(X, fbINX))
        X_addfb = np.stack(X_addfb)     # fb*epoch*channel*T
        X_addfb = np.transpose(X_addfb,(1,0,-2,-1))
        
        fb_coefs = np.expand_dims(np.arange(1, self.n_band+1)**-1.25+0.25, axis=0)
        
        result = []
        corrmatrix = np.zeros((epochNUM, classNUM))
        cca = CCA(n_components=1,max_iter=100000)
        for epochINX, epoch in enumerate(X_addfb):
            for classINX, template in enumerate(rctX):
                rtemp = np.zeros((self.n_band,1))
                for fbINX, (fbepoch, fbtemplate, fbfilter) in enumerate(zip(epoch, template, self.filters)):
                    # u,v = cca.fit_transform(fbtemplate.T,fbepoch.T)
                    u,v = cca.fit_transform(fbtemplate.T,(fbfilter.dot(fbepoch)).T)
                    rtemp[fbINX,:] = np.corrcoef(u.T,v.T)[0,1]
                r=fb_coefs.dot(rtemp)
                corrmatrix[epochINX,classINX]=r
            result.append(np.argmax(corrmatrix[epochINX,:]))
        self.result = np.stack(result)
        self.corrmatrix = corrmatrix
        
        return self.result
        
    def fit_transform(self,X,y):
        winLENs = round(self.winLEN*self.srate)
        trialLength = X.shape[-1]/self.srate
        X = X[:,:,self.latency:self.latency+winLENs]
        X = X-np.mean(X,axis=-1,keepdims=True)
        self.rctX = self.gettemplate(trialLength=trialLength)
        rctX = self.rctX[...,self.latency:self.latency+winLENs]    #fq*fb*class*T
        rctX = np.transpose(rctX,(2,1,0,-1))      #class*fb*fq*T
        
        evokeds = []
        for classINX in np.unique(y):
            X_class = X[y==classINX]
            evokeds.append(np.mean(X_class,axis=0))
        evokeds = np.stack(evokeds)     #class*channel*T
        
        us=[]
        vs=[]
        cca=CCA(n_components=1)
        for classINX, (rct_class, evoked) in enumerate(zip(rctX,evokeds)):
            u,v = cca.fit_transform(rct_class[0].T,evoked.T)
            us.append(u)
            vs.append(v)
        us=np.stack(us)
        vs=np.stack(vs)
        return us, vs   #u是重建响应，v是实际响应
