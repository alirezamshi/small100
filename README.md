SMaLL-100: Introducing Shallow Multilingual Machine Translation Model for Low-Resource Languages
=================

<p align="center">
  <img src="logo.png" width="700"/>
</p>

SMaLL-100 is a compact and fast massively multilingual MT model covering more than 10K language pairs, 
that achieves competitive results with M2M-100 while being much smaller and faster.

We provide the checkpoint in both Fairseq and Huggingface🤗 formats. 

Contents
---------------
- [Demo](#demo)
- [Fairseq](#fairseq)
- [Huggingface🤗](#huggingface)
- [Tokenization + spBLEU](#tokenize)
- [Languages Covered](#languages)
- [Citation](#citation)

<a name="demo"/>  

Demo
--------------  
The demo fo SMaLL-100 model is available [here](https://huggingface.co/spaces/alirezamsh/small100).

<a name="fairseq"/>  

Fairseq
--------------  

You should first install the latest version of Fairseq:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
Please follow [fairseq repo](https://github.com/facebookresearch/fairseq) for further detail.

## Generation with SMaLL-100

1. Download pre-trained model from [here](https://drive.google.com/file/d/1d6Nl3Pbx7hPUbNkIq-c7KBuC3Vi5rR8D/view?usp=sharing) and put it in ```/model``` directory.
2. Pre-process the evaluation set (sample data is provided in ```/data```).
```
fairseq=/path/to/fairseq
cd $fairseq
for lang in af en ; do
    python scripts/spm_encode.py \
        --model model/spm.128k.model \
        --output_format=piece \
        --inputs=data/test.af-en.${lang} \
        --outputs=spm.af-en.${lang}
done

fairseq-preprocess \
    --source-lang af --target-lang en \
    --testpref spm.af-en \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict model/data_dict.128k.txt --tgtdict model/data_dict.128k.txt
```
3. Translate the data by passing the pre-processed input.
```
fairseq-generate \
   data_bin \
   --batch-size 1 \
   --path model/model_small100_fairseq.pt \
   --fixed-dictionary model/model_dict.128k.txt \
   -s af -t en \
   --remove-bpe 'sentencepiece' \
   --beam 5 \
   --task translation_multi_simple_epoch \
   --lang-pairs model/language_pairs_small_models.txt \
   --encoder-langtok tgt \
   --gen-subset test > test.af-en.out
 
 cat test.af-en.out | grep -P "^H" | sort -V | cut -f 3- > test.af-en.out.clean
 
 ```
 
<a name="huggingface"/>  

HuggingFace🤗
-----------------  
First you should install ```transformers``` and ```sentencepiece``` packages:
```
pip install transformers sentencepiece
```

The model architecture and config are the same as [M2M-100](https://huggingface.co/facebook/m2m100_418M) implementation, we just modify the tokenizer to adjust language codes. So, you should load the tokenizer locally from ```tokenization_small100.py``` file.
```
from transformers import M2M100ForConditionalGeneration
from tokenization_small100 import SMALL100Tokenizer

hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
chinese_text = "生活就像一盒巧克力。"

model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

# translate Hindi to French
tokenizer.tgt_lang = "fr"
encoded_hi = tokenizer(hi_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "La vie est comme une boîte de chocolat."

# translate Chinese to English
tokenizer.tgt_lang = "en"
encoded_zh = tokenizer(chinese_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Life is like a box of chocolate."
```
Check the [model hub](https://huggingface.co/alirezamsh/small100) for further details.

<a name="tokenize"/>  

Tokenization + spBLEU
-------------

As mentioned in the paper, we use spBLEU as the MT metric. It uses SentencePiece (SPM) tokenizer with 256K tokens, 
then BLEU is calculated on the tokenized text.
```
git clone --single-branch --branch adding_spm_tokenized_bleu https://github.com/ngoyal2707/sacrebleu.git
cd sacrebleu
python setup.py install
```
To get the score, run:
```
sacrebleu test.af-en.out.ref < test.af-en.out.clean --tokenize spm
```

<a name="languages"/>  

Languages Covered
-------------
Afrikaans (af), Amharic (am), Arabic (ar), Asturian (ast), Azerbaijani (az), Bashkir (ba), Belarusian (be), Bulgarian (bg), Bengali (bn), Breton (br), Bosnian (bs), Catalan; Valencian (ca), Cebuano (ceb), Czech (cs), Welsh (cy), Danish (da), German (de), Greeek (el), English (en), Spanish (es), Estonian (et), Persian (fa), Fulah (ff), Finnish (fi), French (fr), Western Frisian (fy), Irish (ga), Gaelic; Scottish Gaelic (gd), Galician (gl), Gujarati (gu), Hausa (ha), Hebrew (he), Hindi (hi), Croatian (hr), Haitian; Haitian Creole (ht), Hungarian (hu), Armenian (hy), Indonesian (id), Igbo (ig), Iloko (ilo), Icelandic (is), Italian (it), Japanese (ja), Javanese (jv), Georgian (ka), Kazakh (kk), Central Khmer (km), Kannada (kn), Korean (ko), Luxembourgish; Letzeburgesch (lb), Ganda (lg), Lingala (ln), Lao (lo), Lithuanian (lt), Latvian (lv), Malagasy (mg), Macedonian (mk), Malayalam (ml), Mongolian (mn), Marathi (mr), Malay (ms), Burmese (my), Nepali (ne), Dutch; Flemish (nl), Norwegian (no), Northern Sotho (ns), Occitan (post 1500) (oc), Oriya (or), Panjabi; Punjabi (pa), Polish (pl), Pushto; Pashto (ps), Portuguese (pt), Romanian; Moldavian; Moldovan (ro), Russian (ru), Sindhi (sd), Sinhala; Sinhalese (si), Slovak (sk), Slovenian (sl), Somali (so), Albanian (sq), Serbian (sr), Swati (ss), Sundanese (su), Swedish (sv), Swahili (sw), Tamil (ta), Thai (th), Tagalog (tl), Tswana (tn), Turkish (tr), Ukrainian (uk), Urdu (ur), Uzbek (uz), Vietnamese (vi), Wolof (wo), Xhosa (xh), Yiddish (yi), Yoruba (yo), Chinese (zh), Zulu (zu)

TODO
-------------
* Integrate the tokenizer into HuggingFace repo
* Add scripts to automatically run evaluation on low-resource benchmarks


<a name="citation"/>  

Citation
-------------

<a name="citations"/>  

If you use this code for your research, please cite the following work:
```
@article{mohammadshahi2022small,
  title={SMaLL-100: Introducing Shallow Multilingual Machine Translation Model for Low-Resource Languages},
  author={Mohammadshahi, Alireza and Nikoulina, Vassilina and Berard, Alexandre and Brun, Caroline and Henderson, James and Besacier, Laurent},
  journal={arXiv preprint arXiv:2210.11621},
  year={2022}
}
```
Have a question not listed here? Open [a GitHub Issue](https://github.com/alirezamshi/small100/issues) or 
send us an [email](alireza.mohammadshahi@idiap.ch).
