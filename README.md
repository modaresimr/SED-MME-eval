# SED-MME-eval


This tool prepare a multimodal evaluation for Sound Event Detection (SED) systems

Please install Jupyter and open this notebook [SED-MME-eval.ipynb](SED-MME-eval.ipynb) 




# Installation

Please run the following line in the command prompt to install the metric
```
pip install -r requirements.txt
pip install ./
```


# Usage

```
import mme_eval.multi_eval
res1=mme_eval.multi_eval.get_single_result(groundtruthfile,peredictionfile,metadatafile,debug=0)
```

If the groundtruth,perediction,metadata are available in dataframe 
```
res1=mme_eval.multi_eval.get_single_result_df(groundtruth,perediction,metadata,debug=0)
```

