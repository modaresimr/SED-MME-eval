import numpy as np
import pandas as pd
import os
import glob

from mme_eval.other_eval import psds_score, compute_psds_from_operating_points, compute_metrics
from . import other_eval 
import psds_eval
import warnings
 

def psds_metric(dtc_threshold, gtc_threshold, cttc_threshold, ground_truth, metadata, predictions):
        psds = psds_eval.PSDSEval(dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold, cttc_threshold=cttc_threshold, ground_truth=ground_truth, metadata=metadata.reset_index().rename(columns={'index':'filename'}))
        # from psds_macro_f1, psds_f1_classes = psds.compute_macro_f_score(predictions)
        det_t = psds._init_det_table(predictions)
        counts, tp_ratios, _, _ = psds._evaluate_detections(det_t)
        per_class_tp = np.diag(counts)[:-1]
        
        num_gts=np.divide(per_class_tp , tp_ratios, out=np.zeros_like(per_class_tp), where=tp_ratios!=0)
        
        per_class_fp = counts[:-1, -1]
        per_class_fn = num_gts - per_class_tp
        classes=sorted(set(psds.class_names).difference([psds_eval.psds.WORLD]))        

        dic = {c: {'Ntp':tp,'Nfp':fp,'Nfn':fn} for c, tp,fp,fn in zip(classes, per_class_tp,per_class_fp,per_class_fn)}


        return dic

def compute_mme(ground_truth, metadata, predictions ,debug=[]):
    from . import mme as m
    ev= m.eval(ground_truth,predictions,metadata,debug=debug)
    mm=ev[list(ev.keys())[0]].keys()
    out={m:{c:{'Ntp':ev[c][m]['tp'],'Nfp':ev[c][m]['fp'],'Nfn':ev[c][m]['fn'],'Ntn':ev[c][m]['tn']}  for c in ev} for m in mm}
    return out
            
def get_single_result(gtf,pef,metaf=None,psdsf=None,debug=[]):
    res={'macro_avg','micro_avg','class'}

    # gem=computeGem(gtf,pef)          

    groundtruth = pd.read_csv(gtf, sep="\t")
    # Evaluate a single prediction
    predictions = pd.read_csv(pef, sep="\t")
    meta_df=None
    if(metaf is not None):
        meta_df = pd.read_csv(metaf, sep="\t")

    # print(meta_df)
    return get_single_result_df(groundtruth,predictions,meta_df,debug=debug)

def get_single_result_df(groundtruth,predictions,meta_df=None,psdsf=None,debug=[]):
    out={}
    if meta_df is None:
        meta_df=pd.DataFrame(groundtruth.append(predictions).groupby(['filename'])['offset'].max().rename('duration'))
        meta_df[meta_df['duration']<10]=10

    if 'filename'  in meta_df.columns:
        meta_df=meta_df.set_index('filename')
        


    def calcs(metric):
        df=pd.DataFrame(metric).T
        df.loc['micro-avg']=df.sum()
        df['recall']=df['Ntp']/(df['Ntp']+df['Nfn'])
        df['precision']=df['Ntp']/(df['Ntp']+df['Nfp'])
        df['f1']=2*df['precision']*df['recall']/(df['precision']+df['recall'])    
        df.loc['macro-avg']=df.drop('micro-avg').mean()
        # df['f1']=2*df['precision']*df['recall']/(df['precision']+df['recall'])    
        return df
    events_metric = other_eval.event_based_evaluation_df(groundtruth, predictions, t_collar=0.200,percentage_of_length=0.2)
    events_metric_df=calcs(events_metric.class_wise)
    out["collar"]=events_metric_df
    #print('events_metric',events_metric)
    # groundtruth=groundtruth[groundtruth['event_label']=='Blender']
    # predictions=predictions[predictions['event_label']=='Blender']
    segment_metric = other_eval.segment_based_evaluation_df(groundtruth, predictions,meta_df, time_resolution=1.)
    # print(segment_metric.class_wise)
    #print('segment_metric',segment_metric)
    segment_metric_df=calcs(segment_metric.class_wise)

    out["segment"]=segment_metric_df
    #macro_f1_event = events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
    #macro_f1_segment = segment_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

    

    thresh=np.arange(0.1,1,.2)#np.arange(0.1,1,.6)=[0.1,0.7]  ,np.arange(0.1,1,.2)=[0.1, 0.3, 0.5, 0.7, 0.9]
    thresh=[0.1, 0.3, 0.5, 0.8,0.85, 0.9]
    for t in thresh:
        psds=psds_metric(dtc_threshold=t, gtc_threshold=t, cttc_threshold=.3, ground_truth=groundtruth, metadata=meta_df,predictions=predictions)
        psds_df=calcs(psds)
        out[f'psd d/gtc={t}']=psds_df
    
    metadata={}
    for i,f in meta_df.iterrows():
        metadata[i]=(0,f['duration'])

    mme=compute_mme(groundtruth,metadata, predictions,debug=debug)
    for m in mme:
        out[m]=calcs(pd.DataFrame(mme[m]))
    # print(out)
    return out


def array2SED(g,p,duration,path=None):  
    gdf=pd.DataFrame(g,columns=['onset','offset'])
    gdf['event_label']='Test'
    gdf['filename']='Test'
    gdf=gdf[['filename','onset','offset','event_label']]
    pdf=pd.DataFrame(p,columns=['onset','offset'])
    pdf['event_label']='Test'
    pdf['filename']='Test'
    pdf=pdf[['filename','onset','offset','event_label']]
    meta_df=pd.DataFrame(columns=['filename','duration'])
    meta_df['filename']=gdf['filename'].unique()
#     meta_df['duration']=max(gdf['offset'].max(),pdf['offset'].max())
    meta_df['duration']=duration
    if path != None:
        import os
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(f'{path}/meta.tsv',sep='\t',index=False)
        pdf.to_csv(f'{path}/p.tsv',sep='\t',index=False)
        gdf.to_csv(f'{path}/g.tsv',sep='\t',index=False)
    return gdf,pdf,meta_df