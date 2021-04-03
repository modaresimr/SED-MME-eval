# SED-MME-eval


Please install Jupyter and open this notebook [SED-MME-eval.ipynb](SED-MME-eval.ipynb)


# Usage

```
import mme_eval.multi_eval
res1=mme_eval.multi_eval.get_single_result(groundtruthfile,peredictionfile,metadatafile,debug=0)
```

If the groundtruth,perediction,metadata are available in dataframe 
```
res1=mme_eval.multi_eval.get_single_result_df(groundtruth,perediction,metadata,debug=0)
```

