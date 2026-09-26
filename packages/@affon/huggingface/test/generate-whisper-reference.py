"""Small independent feature oracle using Transformers' default PyTorch STFT."""
import json
from pathlib import Path
import numpy as np
from transformers import WhisperFeatureExtractor
p=WhisperFeatureExtractor();cases=[]
for length in [400,1200,4000,480000]:
    wave=(((np.arange(length)*73)%251-125)/256).astype(np.float32)
    features=p(wave,sampling_rate=16000,return_tensors='np').input_features[0]
    indices=[0,1,2,3,10,20,25,2999]
    cases.append({'length':length,'indices':indices,'values':features[:,indices].tolist()})
Path(__file__).with_name('fixtures').joinpath('whisper-processor.json').write_text(json.dumps(cases))
