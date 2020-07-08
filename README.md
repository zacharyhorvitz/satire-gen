# Context-Driven Satirical Headline Generation
Public Repo for [Context-Driven Satirical Headline Generation](https://www.aclweb.org/anthology/2020.figlang-1.5.pdf) (ACL 2020 FigLang Workshop Paper)


## Data
-  [Onion Headline -> Metadata](https://context-driven-satire.s3-us-west-2.amazonaws.com/onion_to_data.json) 

Example:
<code>
"Melting Giraffe Congressman Warns Impeachment Distracting From Surreal Issues": {"text": ["WASHINGTON - Arguing that a protracted congressional trial was..."], "tags": ["impeachment\n"], "date": "1/24/20", "category": "News in Brief", "link": "https://politics.theonion.com/melting-giraffe-congressman-warns-impeachment-distracti-1841207701"}
  </code>

-  [Onion Headline -> Ranked CNN, Wikipedia Paragraphs](https://context-driven-satire.s3-us-west-2.amazonaws.com/raw_headlines_to_ranked_results.json) 

<code>
[Satirical Headline]: {"cnn": [Ranked From CNN (<1 week of article publishing)] ,"wiki": [Ranked from Wikipedia]}
  </code>


## Model
- PreSum Abstractive Summarization Model [[Original Code]](https://github.com/nlpyang/PreSumm) [[Weights]](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)
- SatireGen Context Model [[Weights]](cs.brown.edu/research/satire/context_model_2250)
- SatireGen Decoder-Weighted-Context Model [[Weights]]( cs.brown.edu/research/satire_d_context_model_2000)
- SatireGen Abstraction-Context Model [[Weights]]( cs.brown.edu/research/satire_a_context_model_2000)

## Baselines
- [Pretrained GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
- GPT-2 Satire Baseline [[Weights]]
- GPT-2 News Baseline [[Weights]]


## TODO:
### Include (Move from private repo)
- Clean up superfluous scripts across src folders / rerun each script
- Add training code for baselines
- Add data for baselines/unfun (host on department machine)
- News Retrieval scripts
- Clean up data processing scripts/confirm runthrough of code
- Upload Bertdata (preprocesses) for Training
- Add weight files (Host files on Brown department machine)
- Turker Data/Analysis code
