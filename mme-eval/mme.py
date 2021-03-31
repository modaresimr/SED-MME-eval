import math

import pandas as pd
from intervaltree import intervaltree
from matplotlib.pylab import plt
from pandas.core.frame import DataFrame
from prompt_toolkit.shortcuts import set_title
from wardmetrics.core_methods import eval_events, merge_events_if_necessary
import numpy as np
import result_analyse.SpiderChart as spiderchart


def eval(gt, pt,meta=None, clas=None,debug=0,calcne=0):
    #%matplotlib inline
    if clas is None:
        clas=gt.event_label.append(pt.event_label).unique()
    
    result={}
    for c in clas:
        gtc=gt.loc[gt.event_label==c]
        ptc=pt.loc[pt.event_label==c]
        
        for f in gtc.filename.append(ptc.filename).unique():
            # if(c=='Blender'):debug=1
            if debug:print(f'============== class={c}=========file={f}')
            g=gtc.loc[gtc.filename==f][['onset','offset']].values
            p=ptc.loc[ptc.filename==f][['onset','offset']].values
            m=(0,10)
            if meta is not None and f in meta: m=meta[f]
            try:
                # debug=c=='Dishes'
                out=eval_my_metric(g,p,m,debug=debug,calcne=calcne)
            except Exception as e:
                print(f'============== class={c}=========file={f}')
                print(e)
                out=eval_my_metric(g,p,m,debug=1,calcne=calcne)

                raise
            if (c not in result):
                result[c]=out
            else:
                for m in out:
                    for t in out[m]:
                        result[c][m][t]+=out[m][t]

    return result


def intersection(e1,e2):    
    inter=[max(e1[0],e2[0]),min(e1[1],e2[1])]
    if(inter[1]<=inter[0]): 
        inter=None
#     print(e1,e2,inter)
    return inter

def dur(e):
    d= e[1]-e[0]
    if(d<0):
        raise Exception('erorr duration is less than zero')
    return d

def Z(rel,e,X,Y):
    s={}
    for e2 in rel[X][e][Y]:
        for e in rel[Y][e2][X]:
            s[e]=1
    
    return len(s)

def eval_my_metric(real,pred,duration=(0,10),alpha=2,debug=0,calcne=1):
        # if debug:debug={'D':1, 'M':1,'U':1, 'T':1, 'R':1,'B':1,'V':1}#V:verbose
        if debug:debug={'D':0, 'M':0,'U':0, 'T':0, 'R':0,'B':0,'V':1}
        else:debug={'D':0, 'M':0, 'U':0,'T':0, 'R':0,'B':0,'V':0}
        

        # real=merge_events_if_necessary(real)
        # pred=merge_events_if_necessary(pred)
        # real_tree=_makeIntervalTree(real,'r')
        # pred_tree=_makeIntervalTree(pred,'p')
        # duration=(min(duration[0],real[0][0]),max(duration[1],real[-1][1]))
        real=real[real[:,0].argsort(),:]
        pred=pred[pred[:,0].argsort(),:]
        real=np.vstack((real,[duration[1],duration[1]]))# add a zero duration event in the end for ease comparision the last event
        pred=np.vstack((pred,[duration[1],duration[1]]))# add a zero duration event in the end for ease comparision the last event
        #_ means negative
        rel={'r+':{},'r-':{},'p+':{},'p-':{}}
        r_0=(duration[0],real[0][0])
        r_n=(real[-1][1],duration[1])
        metric={}
        pi=0
        rcalc=[]
        real_=[]
        pred_=[]
        ri_=0
        for ri in range(len(real)):
            r=real[ri]
            rp=real[ri-1] if ri>0 else [duration[0],duration[0]]
            r_=(rp[1],r[0])
            
            tmpr={'p+':{},'p-':{}}
            tmpr_={'p+':{},'p-':{}}
            rel['r+'][ri]=tmpr
            
            if(dur(r_)>0):
                real_.append(r_)
                ri_=len(real_)-1
                rel['r-'][ri_]=tmpr_
            
            cond=pi<len(pred) 
            pi_=-1

            while  cond:
                pp=pred[pi-1] if pi>0 else [duration[0],duration[0]]
                p=pred[pi]
                p_=(pp[1],p[0])
                
                if(dur(p_)>0 and (len(pred_)==0 or pred_[-1]!=p_)):
                    pred_.append(p_)
                pi_=len(pred_)-1

                if not(pi in rel['p+']):
                    rel['p+'][pi]={'r+':{},'r-':{}}
                if not(pi_ in rel['p-']):
                    rel['p-'][pi_]={'r+':{},'r-':{}}
                tmpp=rel['p+'][pi]
                tmpp_=rel['p-'][pi_]

                

                rinter  =intersection(r,p)
                rinter_ =intersection(r,p_)
                r_inter =intersection(r_,p)                
                r_inter_=intersection(r_,p_)
                if(rinter is not None):
                    # tmpr['p+'].append((pi,rinter))
                    # tmpp['r+'].append((ri,rinter))
                    tmpr['p+'][pi]=rinter
                    tmpp['r+'][ri]=rinter
                if(rinter_ is not None):
                    # tmpr['p-'].append((pi,rinter_))
                    # tmpp_['r+'].append((ri,rinter_))
                    tmpr['p-'][pi_]=rinter_
                    tmpp_['r+'][ri]=rinter_
                if(r_inter is not None):
                    # tmpr_['p+'].append((pi,r_inter))
                    # tmpp['r-'].append((ri,r_inter))
                    tmpr_['p+'][pi]=r_inter
                    tmpp['r-'][ri_]=r_inter
                if(r_inter_ is not None):
                    # tmpr_['p-'].append((pi,r_inter_))
                    # tmpp_['r-'].append((ri,r_inter_))
                    tmpr_['p-'][pi_]=r_inter_
                    tmpp_['r-'][ri_]=r_inter_
                
                
                if pred[pi][1] < r[1]:
                    pi+=1
                else: cond=False
                
            
            # for k in list(rel.keys()):
            #     if len(rel[k])>0: continue
            #     del rel[k]
        
        real=np.delete(real,-1,0)#real.pop()
        pred=np.delete(pred,-1,0)#pred.pop()
#         if(dur(pred_[-1])==0):pred_=np.delete(pred_,-1,0)
#         if(dur(real_[-1])==0):real_=np.delete(real_,-1,0)

        
        out={
            'detection':        {'tp':0,'fp':0,'fn':0,'tn':0},
            'detect-mono':      {'tp':0,'fp':0,'fn':0,'tn':0},
            'monotony':         {'tp':0,'fp':0,'fn':0,'tn':0},
            'uniformity':       {'tp':0,'fp':0,'fn':0,'tn':0},
            'total duration':   {'tp':0,'fp':0,'fn':0,'tn':0},
            'relative duration':{'tp':0,'fp':0,'fn':0,'tn':0},
            'boundary onset':   {'tp':0,'fp':0,'fn':0,'tn':0},
            'boundary offset':  {'tp':0,'fp':0,'fn':0,'tn':0},
        }
        
        if debug['V']:
            print("real=",real)
            print("pred=",pred)
            print("real_=",real_)
            print("pred_=",pred_)
            #for x in rel:
            [print(f'{x}: {rel[x]}') for x in rel]
        if debug['V']:plot_events(real,pred,duration,real_,pred_)
        for ri in range(len(real)):
            tpd=int(len(rel['r+'][ri]['p+'])>0)
            out['detect-mono']['tp']+=tpd
            out['detection']['tp']+=tpd
            if debug['D']: print(f" D TP+{tpd}      ri={ri}, p+={rel['r+'][ri]['p+']}>0")
            #monotony {
            
            if (len(rel['r+'][ri]['p+'])==1):
                for rpi in rel['r+'][ri]['p+']:
                    if len(rel['p+'][rpi]['r+'])==1:
                        out['monotony']['tp']+=1
                        if debug['M']:print(f"  M TP+1     rel[r+][{ri}][p+]={rel['r+'][ri]['p+']}==1 rel[p+][{rpi}][r+]={rel['p+'][rpi]['r+']}==1")
                    elif(len(rel['p+'][rpi]['r+'])==0):
                        print('error it can not be zero')
                    elif debug['M']:print(f"  M--tp rel[r+][{ri}][p+]={rel['r+'][ri]['p+']}==1 rel[p+][{rpi}][r+]={rel['p+'][rpi]['r+']}>1")
            #}
            
            for pi in rel['r+'][ri]['p+']:
                tpt=dur(rel['r+'][ri]['p+'][pi])
                tpr=tpt/dur(real[ri])
                out['total duration']['tp']+=tpt
                out['relative duration']['tp']+=tpr
                if debug['T']:print(f"   T tp+={tpt}             rel[r+][{ri}][p+][{pi}]=dur({rel['r+'][ri]['p+'][pi]})")
                if debug['R']:print(f"    R tp+={tpr}             rel[r+][{ri}][p+][{pi}]==dur({rel['r+'][ri]['p+'][pi]}) / real[{ri}]=dur({real[ri]})")

            ####Uniformity{
            tpuc=Z(rel,ri,'r+','p+')
            tpu=1/tpuc if tpuc>0 else 0
            if calcne or tpuc>0:
                out['uniformity']['tp']+=tpu
                out['uniformity']['fn']+=1-tpu
                if debug['U']:print(f"  U tp+{tpu}  fn+{1-tpu}           Z[r+][{ri}][p+]=={tpuc}")
            ####Uniformity}

            rps=list(rel['r+'][ri]['p+'].keys())
            ####################boundary onset
            if(len(rps)==0):
                if calcne:
                    out['boundary onset']['fn']+=1
                    out['boundary offset']['fn']+=1
                    if debug['B']:print(f"     B onset&offset fn+{1}  ri={ri} pi=[]          ")
            else:
                relp=pred[rps[0]]
                boundry_error_b=real[ri][0]-relp[0]
                ufbp=max(0,-boundry_error_b)/dur(real[ri])
                ofbp=min(1,max(0,boundry_error_b)/dur(real[ri]))
                tpbp=min(1,max(0,1-ufbp-ofbp))
                out['boundary onset']['tp']+=tpbp
                out['boundary onset']['fn']+=ufbp
                out['boundary onset']['fp']+=ofbp
                if debug['B']:print(f"     B onset tp+{tpbp} fp+{ofbp} fn+{ufbp}  ri={ri} pi={rps[0]}     boundary_error_onset={boundry_error_b}     ")

                #####################boundary offset
                relp=pred[rps[-1]]
                boundry_error_e=relp[1]-real[ri][1]
                ufep=min(1,max(0,-boundry_error_e)/dur(real[ri]))
                ofep=min(1,max(0,boundry_error_e)/dur(real[ri]))
                tpep=max(0,1-ufep-ofep)
                out['boundary offset']['tp']+=tpep
                out['boundary offset']['fn']+=ufep
                out['boundary offset']['fp']+=ofep
                if debug['B']:print(f"     B offset tp+{tpep} fp+{ofep} fn+{ufep}  ri={ri} pi={rps[-1]}     boundary_error_onset={boundry_error_e}     ")

            

            for pi in rel['r+'][ri]['p-']:
                fnt=dur(rel['r+'][ri]['p-'][pi])
                fnr=fnt/dur(real[ri])
                out['total duration']['fn']+=fnt
                out['relative duration']['fn']+=fnr if fnr<0.99 else calcne
                if debug['T']:print(f"   T fn+={fnt}             rel[r+][{ri}][p-][{pi}]=dur({rel['r+'][ri]['p-'][pi]})")
                if debug['R']:print(f"    R fn+={fnr}             rel[r+][{ri}][p-][{pi}]==dur({rel['r+'][ri]['p-'][pi]}) / real[{ri}]=dur({real[ri]})")
                

        for ri in range(len(real_)):
            tnd=int(len(rel['r-'][ri]['p-'])>0)
            out['detect-mono']['tn']+=tnd
            out['detection']['tn']+=tnd
            if debug['D']: print(f" D TN+{tnd}      ri-={ri}, p-={rel['r-'][ri]['p-']}>0")
            #monotony {
            
            if (len(rel['r-'][ri]['p-'])==1):
                for rpi in rel['r-'][ri]['p-']:
                    if len(rel['p-'][rpi]['r-'])==1:
                        out['monotony']['tn']+=1
                        if debug['M']:print(f"  M TN+1     rel[r-][{ri}][p-]={rel['r-'][ri]['p-']}==1 rel[p-][{rpi}][r-]={rel['p-'][rpi]['r-']}==1")
                    elif(len(rel['p-'][rpi]['r-'])==0):
                        print('error it can not be zero')
                    elif debug['M']:print(f"  M--tn rel[r-][{ri}][p-]={rel['r-'][ri]['p-']}==1 rel[p-][{rpi}][r-]={rel['p-'][rpi]['r-']}>1")
            #}

            for pi in rel['r-'][ri]['p-']:
                tnt=dur(rel['r-'][ri]['p-'][pi])
                tnr=tnt/dur(real_[ri])
                out['total duration']['tn']+=tnt
                out['relative duration']['tn']+=tnr
                if debug['T']:print(f"   T tn+={tnt}             rel[r-][{ri}][p-][{pi}]=dur({rel['r-'][ri]['p-'][pi]})")
                if debug['R']:print(f"    R tn+={tnr}             rel[r-][{ri}][p-][{pi}]==dur({rel['r-'][ri]['p-'][pi]}) / real_[{ri}]=dur({real_[ri]})")
            for pi in rel['r-'][ri]['p+']:
                fpt=dur(rel['r-'][ri]['p+'][pi])
                fpr=fpt/dur(real_[ri])
                out['total duration']['fp']+=fpt
                out['relative duration']['fp']+=fpr if fpr<0.99 else calcne
                if debug['T']:print(f"   T fp+={fpt}             rel[r-][{ri}][p+][{pi}]=dur({rel['r-'][ri]['p+'][pi]})")
                if debug['R']:print(f"    R fp+={fpr}             rel[r-][{ri}][p+][{pi}]==dur({rel['r-'][ri]['p+'][pi]}) / real_[{ri}]=dur({real_[ri]})")

            if 0:
                rps=list(rel['r-'][ri]['p-'].keys())
                ####################boundary onset
                if(len(rps)==0):
                    out['boundary onset']['fp']+=1
                    out['boundary offset']['fp']+=1
                    if debug['B']:print(f"     B onset&offset fp+{1}  ri-={ri} pi-=[]          ")
                else:
                    relp=pred_[rps[0]]
                    boundry_error_b=real_[ri][0]-relp[0]
                    ufbp=min(1,max(0,-boundry_error_b)/dur(real_[ri]))
                    ofbp=min(1,max(0,boundry_error_b)/dur(real_[ri]))
                    tpbp=max(0,1-ufbp-ofbp)
                    out['boundary onset']['tn']+=tpbp
                    out['boundary onset']['fn']+=ofbp
                    out['boundary onset']['fp']+=ufbp
                    if debug['B']:print(f"     B onset tn+{tpbp} fp+{ufbp} fn+{ofbp}   ri-={ri} pi-={rps[0]}    boundary_error_onset={boundry_error_b}     ")

                    #####################boundary offset
                    relp=pred_[rps[-1]]
                    boundry_error_e=relp[1]-real_[ri][1]
                    ufep=min(1,max(0,-boundry_error_e)/dur(real_[ri]))
                    ofep=min(1,max(0,boundry_error_e)/dur(real_[ri]))
                    tpep=max(0,1-ufep-ofep)
                    out['boundary offset']['tn']+=tpep
                    out['boundary offset']['fn']+=ofep
                    out['boundary offset']['fp']+=ufep
                    if debug['B']:print(f"     B offset tn+{tpep} fp+{ufep} fn+{ofep}   ri-={ri} pi-={rps[-1]}    boundary_error_onset={boundry_error_e}     ")


        out['detect-mono']['fp']=len(real_)-out['detect-mono']['tn']
        if debug['D']: print(f" D fp={out['detect-mono']['fp']} #r-={len(real_)} - tn={out['detect-mono']['tn']}"  )
        out['detect-mono']['fn']=len(real)-out['detect-mono']['tp']
        out['detection']['fn']=len(real)-out['detection']['tp']
        if debug['D']: print(f" D fn={out['detect-mono']['fn']} #r+={len(real)} - tp={out['detect-mono']['tp']}"  )
                        
        out['monotony']['fn']=len(real)-out['monotony']['tp']#+len(pred_)-out['monotony']['tn']
        if debug['M']: print(f"  M fn={out['monotony']['fn']}     #r+={len(real)} - tp={out['monotony']['tp']} //+ #p-={len(pred_)} - tn={out['monotony']['tn']}")
        out['monotony']['fp']=len(pred)-out['monotony']['tp']#+len(real_)-out['monotony']['tn']
        if debug['M']: print(f"  M fp={out['monotony']['fp']}     #p+={len(pred)} - tp={out['monotony']['tp']} //+ #r-={len(real_)} - tn={out['monotony']['tn']}")
        
                        
        for pi in range(len(pred)):
            fpd=int(len(rel['p+'][pi]['r+'])==0)
            out['detect-mono']['fp']+=fpd
            out['detection']['fp']+=fpd
            if debug['D']: print(f" D FP+{fpd}      pi={pi}, r={rel['p+'][pi]['r+']}==0")
            ####Uniformity{
            fpuc=Z(rel,pi,'p+','r+')
            if calcne or fpuc>0:
                fpu=1- (1/fpuc if fpuc>0 else 0)
                out['uniformity']['fp']+=fpu
                if debug['U']:print(f"  U fp+{fpu}             Z[p+][{pi}][r+]=={fpuc}")
            ####Uniformity}
#             for ri in rel['p+'][pi]['r-']:
#                 out['total duration']['fp']+=dur(rel['p+'][pi]['r-'][ri])
#                 out['relative duration']['fp']+=dur(rel['p+'][pi]['r-'][ri])/dur(pred[pi])

        for pi in range(len(pred_)):
            fnd=int(len(rel['p-'][pi]['r-'])==0)
            out['detect-mono']['fn']+=fnd
            if debug['D']: print(f" D FN+{fnd}      pi-={pi}, r-={rel['p-'][pi]['r-']}==0")
#             for ri in rel['p-'][pi]['r+']:
#                 out['total duration']['fn']+=dur(rel['p-'][pi]['r+'][ri])
#                 out['relative duration']['fn']+=dur(rel['p-'][pi]['r+'][ri])/dur(pred_[pi])

#         plot_events_with_event_scores(range(len(real)),range(len(pred)),real,pred)
#         plot_events_with_event_scores(range(len(real_)),range(len(pred_)),real_,pred_)
        if debug['V']:
            for m in out: print(m,out[m])
        return  out

def plot_events(real,pred,meta,real_,pred_, label=None):
    from matplotlib.pylab import plt
    import random
    fig,ax = plt.subplots(figsize=(15, .8))
    ax.set_title(label)
    plt.xlim(0,max(meta[1],10))
    ax.set_xticks(np.arange(0,max(real[-1][1],10),.1),minor=True)
    maxsize=20
# random.random()/4
    for i in range(min(maxsize,len(pred_))):
        d = pred_[i]
        plt.axvspan(d[0], d[1], 0, 0.6,linewidth=0,edgecolor='k',facecolor='#edb4b4', alpha=.6)
        plt.text((d[1] + d[0]) / 2, 0.2,f'{i}' , horizontalalignment='center', verticalalignment='center')
    for i in range(min(maxsize,len(pred))):
        d = pred[i]
        plt.axvspan(d[0], d[1], 0.0, 0.6,linewidth=0,edgecolor='k',facecolor='#a31f1f', alpha=.6)
        plt.text((d[1] + d[0]) / 2, 0.2,f'{i}' , horizontalalignment='center', verticalalignment='center')
#     maxsize=len(real)
    for i in range(min(maxsize,len(real_))):
        gt = real_[i]
        plt.axvspan(gt[0], gt[1], 0.4, 1,linewidth=0,edgecolor='k',facecolor='#d2f57a', alpha=.6)
        plt.text((gt[1] + gt[0]) / 2, 0.8,f'{i}' , horizontalalignment='center', verticalalignment='center')
        
    for i in range(min(maxsize,len(real))):
        gt = real[i]
        plt.axvspan(gt[0], gt[1], 0.4, 1,linewidth=0,edgecolor='k',facecolor='#1fa331', alpha=.6)
        plt.text((gt[1] + gt[0]) / 2, 0.8,f'{i}' , horizontalalignment='center', verticalalignment='center')
    # plt.grid(True)
    plt.minorticks_on()
    ax.set(yticks=[.25,.75], yticklabels=['P','R'])
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()