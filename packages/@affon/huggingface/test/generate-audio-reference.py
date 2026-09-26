"""Independent AST feature oracle: Transformers 4.56.2 NumPy fallback (no torchaudio)."""
import json
from pathlib import Path
import numpy as np
from transformers import ASTFeatureExtractor
p = ASTFeatureExtractor(max_length=4, num_mel_bins=8, mean=-6.845978, std=5.5654526)
cases=[]
for length in [400, 720, 1600]:
    samples = (((np.arange(length)*73)%251-125)/256).astype(np.float32)
    features = p(samples, sampling_rate=16000, return_tensors='np').input_values.tolist()
    cases.append({'length':length,'features':features})
cases.append({'length':400,'silence':True,'features':p(np.zeros(400,np.float32),sampling_rate=16000,return_tensors='np').input_values.tolist()})
Path(__file__).with_name('fixtures').joinpath('ast-processor.json').write_text(json.dumps(cases))
