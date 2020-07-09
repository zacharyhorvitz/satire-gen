# Context-Driven Satirical Headline Generation
Public Repo for [Context-Driven Satirical Headline Generation](https://www.aclweb.org/anthology/2020.figlang-1.5.pdf) (ACL 2020 FigLang Workshop Paper)

## Example Generations (data/generations)

<pre><code>CONTEXT: [CLS] a 2014 study of the effects of the oil spill on blue ##fin tuna funded by national oceanic and atmospheric administration - ##lr ##b ##- no ##aa - ##rr ##b ##- , stanford university , and the monterey bay aquarium and published in the journal science , found that the toxin ##s from oil spill ##s can cause irregular heartbeat ##s leading to cardiac arrest . calling the vicinity of the spill ` ##` one of the most productive ocean ecosystems in the world ' ##' , the study found that even at very low concentrations ` ##` pa ##h card ##iot ##ox ##ici ##ty was potentially a common form of injury among a broad range of species in the vicinity of the oil . ' ##' another peer ##- ##re ##view ##ed study , released in march 2014 and conducted by 17 scientists from the united states and australia and published in the proceedings of the national academy of sciences , found that tuna and amber ##jack that were exposed to oil from the spill developed def ##or ##mit ##ies of the heart and other organs that would be expected to be fatal or at least life ##- ##sho ##rte ##ning . the scientists said that their findings would most likely apply to other large predator fish and ` ##` even to humans , whose developing hearts are in many ways similar . ' ##' in may 2010 , a local native set up a network for people to volunteer their assistance in cleaning up beaches . boat captains were given the opportunity to offer the use of their boat to help clean and prevent the oil from further spreading . to assist with the efforts the captains had to register their ships with the vessels of opportunity , however an issue arose when more boats registered than actually participated in the clean up efforts - only a third of the registered boats . many local supporters were disappointed with bp ' ##s slow response , prompting the formation of the florida key environmental coalition . on 4 september 2014 , u ##. ##s ##. district judge carl barbie ##r ruled bp was guilty of gross negligence and will ##ful misconduct . he described bp ' ##s actions as ` ##` reckless . ' ##' he said trans ##oc ##ean ' ##s and hall ##ib ##urt ##on ' ##s actions were ` ##` ne ##gli ##gent . ' ##' he app ##ort ##ioned 67 % of the blame for the spill to bp , 30 % to trans ##oc ##ean , and 3 % to hall ##ib ##urt ##on [SEP]

study finds majority of americans still in oil spill [unused1]

CONTEXT: [CLS] according to he ##cht biographer , william mac ##ada ##ms , ` ##` at dawn on sunday , february 20 , 1939 , david se ##lz ##nick ... and director victor fleming shook he ##cht awake to inform him he was on loan from mgm and must come with them immediately and go to work on gone with the wind , which se ##lz ##nick had begun shooting five weeks before . it was costing se ##lz ##nick $ 50 ##, ##00 ##0 each day the film was on hold waiting for a final screenplay re ##write and time was of the essence . he ##cht was in the middle of working on the film at the circus for the marx brothers . recalling the episode in a letter to screenwriter friend gene fowler , he said he had n ##' ##t read the novel but se ##lz ##nick and director fleming could not wait for him to read it . he ##cht left by plane from new york city to hollywood on monday , november 2 , his position on the film already confirmed . once in california , he interviewed 200 girls and 150 men , in order to find the twelve girls and six men necessary for the dance numbers . he ##cht hired frances grant , fresh from assisting larry ce ##ball ##os at fan ##chon and marco , to help him with the new routines on chi chi and her papa ##s . but a week after he ##cht ' ##s arrival , the film was put on hold . to compose the score , se ##lz ##nick chose max steiner , with whom he had worked at r ##ko pictures in the early 1930s . warner bros ##. - ##- who had contracted steiner in 1936 - ##- agreed to lend him to se ##lz ##nick . steiner spent twelve weeks working on the score , the longest period that he had ever spent writing one , and at two hours and thirty ##- ##si ##x minutes long it was also the longest that he had ever written . five orchestra ##tors were hired , including hugo fried ##hof ##er , maurice de pack ##h , bernard ka ##un , adolph de ##uts ##ch and reginald bassett [SEP]

man has n't read his own book [unused1]

CONTEXT: [CLS] the political crisis of 1988 - ##- 89 was testimony to both the party ' ##s strength and its weakness . in the wake of a succession of issues - ##- the pushing of a highly unpopular consumer tax through the diet in late 1988 , the recruit insider trading scandal , which tainted virtually all top ld ##p leaders and forced the resignation of prime minister takes ##hita no ##bor ##u in april - ##lr ##b ##- a successor did not appear until june - ##rr ##b ##- , the resignation in july of his successor , uno [UNK] , because of a sex scandal , and the poor showing in the upper house election - ##- the media provided the japanese with a detailed and embarrassing di ##sse ##ction of the political system . by march 1989 , popular support for the takes ##hita cabinet as expressed in public opinion polls had fallen to 9 percent . uno ' ##s scandal , covered in magazine interviews of a ` ##` kiss and tell ' ##' ge ##isha , aroused the fury of female voters . this is a list of war apology statements issued by the state of japan with regard to the war crimes committed by the empire of japan during world war ii . the statements were made at and after the end of world war ii in asia , from the 1950s to the 2010s . hash ##imo ##to became a key figure in the strong ld ##p faction founded by ka ##ku ##ei tanaka in the 1970s , which later fell into the hands of no ##bor ##u takes ##hita , who then was tainted by the recruit scandal of 1988 . in 1991 , the press had discovered that one of hash ##imo ##to ' ##s secretaries had been involved in an illegal financial dealing . hash ##imo ##to retired as minister of finance from the second kai ##fu cabinet . following the collapse of the bubble economy , the ld ##p momentarily lost power in 1993 ##/ ##9 ##4 during the ho ##so ##kawa and hat ##a anti ##- ##ld ##p coalition cabinets negotiated by ld ##p defect ##or [UNK] oz ##awa [SEP]

japan 's prime minister to be laid off [unused1]
</code></pre>


## Data

-  [Onion Headline -> Ranked CNN, Wikipedia Paragraphs](https://context-driven-satire.s3-us-west-2.amazonaws.com/raw_headlines_to_ranked_results.json) 

<code>
[Satirical Headline]: {"cnn": [Ranked From CNN (<1 week of article publishing)] ,"wiki": [Ranked from Wikipedia]}
  </code>
  

-  [Onion Headline -> Metadata](https://context-driven-satire.s3-us-west-2.amazonaws.com/onion_to_data.json) 

Example:
<code>
"Melting Giraffe Congressman Warns Impeachment Distracting From Surreal Issues": {"text": ["WASHINGTON - Arguing that a protracted congressional trial was..."], "tags": ["impeachment\n"], "date": "1/24/20", "category": "News in Brief", "link": "https://politics.theonion.com/melting-giraffe-congressman-warns-impeachment-distracti-1841207701"}
  </code>



## Model
- PreSumm Abstractive Summarization Model [[Original Code]](https://github.com/nlpyang/PreSumm) [[Weights]](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)
- SatireGen Encoder-Weighted-Context Model [[Weights]](cs.brown.edu/research/satire/context_model_2250)
- SatireGen Decoder-Weighted-Context Model [[Weights]]( cs.brown.edu/research/satire_d_context_model_2000)
- SatireGen Abstraction-Context Model [[Weights]]( cs.brown.edu/research/satire_a_context_model_2000)

To generate satirical headlines you can follow the [original instructions](https://github.com/nlpyang/PreSumm) provided by the PreSumm authors. Specifically, you can:

<pre><code>python  -W ignore::UserWarning  context_satire/PreSumm/src_satire_gen/train.py -task abs -mode test_text -log_file log.txt -sep_optim true -use_interval true -visible_gpus 0  -max_pos 512 -max_length 60 -alpha 0.95 -min_length 5 -result_path results.txt -test_from [MODEL] -text_src [INPUT TEXT DOCUMENT]
</code></pre>

## Baselines
- [Pretrained GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
- GPT-2 Satire Baseline [[Weights]]
- GPT-2 News Baseline [[Weights]]

## TODO:
### Include (Move from private repo)
- Clean up superfluous scripts across src folders / rerun each script
- Add training code for baselines
- Add data for baselines/unfun (host on department machine)
- News retrieval scripts
- Clean up data processing scripts/confirm runthrough of code
- Upload Bertdata (preprocesses) for Training
- Add weight files (Host files on Brown department machine)
