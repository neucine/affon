"""Synthetic offline processor fixtures: tokenizers 0.22.2, Pillow 12.0.0."""
import json
from pathlib import Path
import numpy as np
import tokenizers
import PIL
from PIL import Image
from tokenizers import Tokenizer, AddedToken, models, normalizers, pre_tokenizers, processors

assert tokenizers.__version__ == '0.22.2' and PIL.__version__ == '12.0.0'
root = Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
vocab = {text: i for i, text in enumerate(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'world', 'cafe', '!', ',', '?', '中', '文', 'a', 'short', 'sentence', '.', 'hi', '##s', 'ο', '##σ', '##α'])}
tokenizer = Tokenizer(models.WordPiece(vocab, unk_token='[UNK]'))
tokenizer.normalizer = normalizers.BertNormalizer()
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
tokenizer.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']])
spec = json.loads(tokenizer.to_str())
texts = ['', 'Hello world!', 'CAFÉ, hello?', '中中文', 'hi\x00\ufffd\x01 world', 'CAFE\u0301\t\nhello', '[MASK]Hello', 'Ος ΟΣ ΟΣΑ', 'hello$world', 'a\u00a0short sentence.']
raw = [{'text': t, 'ids': tokenizer.encode(t, add_special_tokens=False).ids} for t in texts]
tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
batches = []
for items in [['Hello world!', 'hi'], ['', 'CAFÉ, hello?'], [('Hello', 'world!'), ('hi', '')]]:
    encoded = tokenizer.encode_batch(items)
    batches.append({'texts': [x[0] for x in items] if isinstance(items[0], tuple) else items,
                    'pairs': [x[1] for x in items] if isinstance(items[0], tuple) else None,
                    'expected': {'input_ids': [x.ids for x in encoded], 'token_type_ids': [x.type_ids for x in encoded], 'attention_mask': [x.attention_mask for x in encoded]}})
(root / 'bert-processor-reference.json').write_text(json.dumps({'oracle': 'tokenizers==0.22.2', 'spec': spec, 'raw': raw, 'batches': batches}, ensure_ascii=False, indent=2) + '\n')
fixture = root / 'processor-fixture'
fixture.mkdir(exist_ok=True)
(fixture / 'tokenizer.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
(fixture / 'tokenizer_config.json').write_text('{"pad_token":"[PAD]","padding_side":"right"}\n')
cases = []
for h,w,oh,ow in [(3,5,7,9),(9,8,3,4),(1,1,4,7),(1,9,3,2),(7,1,3,5),(3,5,3,5),(9,8,9,3),(8,9,2,9)]:
    y,x,c=np.indices((h,w,3));rgb=((x*63+y*37+c*91)%256).astype(np.uint8)
    expected=np.asarray(Image.fromarray(rgb).resize((ow,oh),Image.Resampling.BILINEAR)).tolist()
    cases.append({'input':rgb.tolist(),'height':oh,'width':ow,'expected':expected})
(root / 'resize-reference.json').write_text(json.dumps({'oracle': 'Pillow==12.0.0', 'cases': cases},separators=(',',':')) + '\n')
