# lexical_simplification

## Sentence retrieval

N_words=1
MWEfile=tonikaku
monofile=/lt/scratch/twada/OSCAR/ja_dedup_jumanpp_rc_tok
folder=Folder
mkdir $folder
python extract_sentence.py -MWEfile ${MWEfile} -monofile ${monofile} -N_words ${N_words} -folder $folder

## Sentence clustering

sent=/lt/scratch/twada/lexsub_decontextualised/tsar2022_en_tgtwords/tsar2022_en_tgtwords.txt_silversent_removed.pkl
clustering=kmeans4
echo $clustering
N_sample=300
folder=$(basename "$model")_${n_mask}_MASK_${clustering}_vec_NEW_nofilter
CUDA_VISIBLE_DEVICES=6 python clustering.py -max_tokens 2048  -clustering ${clustering} -N_sample ${N_sample}  -folder $folder -model ${model} -silver_sent ${sent} -n_mask 0 -mask_opt two

## Generate candidates based on the target context
```
model=neuralmind/bert-large-portuguese-cased
tgt_sent=/lt/scratch/twada/simplification/tsar2022_pt_test_mask.txt
model_basename=/lt/scratch/twada/simplification/bert-large-portuguese-cased_0_MASK_kmeans4_vec_NEW
val=0.6
x=20k
folder=tsar2022_pt_TEST_mask_bert_${val}${x}
vec="${model_basename}/K4/K0/vec.txt ${model_basename}/K4/K1/vec.txt ${model_basename}/K4/K2/vec.txt ${model_basename}/K4/K3/vec.txt"
wordlist=/lt/scratch/twada/OSCAR/huggingface_pt.words_txt.lower${x}
CUDA_VISIBLE_DEVICES=0 python generate_.py -wordlist ${wordlist} -lang pt -fasttext ${val} -print_info -folder ${folder} -model ${model} -vec ${vec}  -tgt_sent ${tgt_sent} -lev 0.5 -beam_size 50
```

## Generate candidates with context augmentation
```
cluster_model=neuralmind/bert-large-portuguese-cased
model=google/mt5-large
phrase_list=tsar2022_pt_tgtwords.txt
clustering=kmeans4
echo $clustering
N_sample=300
cluster_folder=bert-large-portuguese-cased_0_MASK_kmeans4_vec_NEW_nofilter_tsar2022_pt_tgtwords
clustered_sents="${cluster_folder}/sents_by_cluster4.pkl"
folder=${cluster_folder}_weight${weight}_$(basename "$model")_bs20_nolenpena
beam_size=20
weight=0
CUDA_VISIBLE_DEVICES=0 python generate_t5.py -phrase_list ${phrase_list} -weight ${weight} -clustered_sents ${clustered_sents} -num_beams ${beam_size} -folder ${folder} -model ${model} -n_mask 1 -max_tokens 4096
```

## Merge candidates 
```

data=test
val=0.6
emb=/lt/scratch/twada/plus_lexsub/tsar2022_pt_test_mask_bert_${val}20k/neuralmind_bert-large-portuguese-cased_beam_50lambda_val0.7_candidates2cossim_score.pkl
folder=bert-large-portuguese-cased_0_MASK_kmeans4_vec_NEW_nofilter_tsar2022_pt_tgtwords_weight_mt5-large_bs20_nolenpena
t5=${folder}/1MASKs_candidates2inner_score.pkl
tgt_sent="TSAR-2022-Shared-Task/datasets/${data}/tsar2022_pt_${data}_none.tsv"
lang=pt
python mix_candidates.py -t5 $t5 -emb $emb -lang ${lang} -folder test_fixed_pt -tgt_sent ${tgt_sent}
```

## Calculate the embedding-similarity ranking
```
val=0.6
folder=/lt/scratch/twada/plus_lexsub/tsar2022_pt_TEST_mask_bert_${val}20k
candidates=${folder}/final_candidates.pkl
tgt_sent=tsar2022_pt_test_mask.txt
model=neuralmind/bert-large-portuguese-cased
CUDA_VISIBLE_DEVICES=0 python reranking.py -fasttext ${val} -lang pt -candidates ${candidates} -folder ${folder} -model ${model} -tgt_sent ${tgt_sent}
```

## Calculate the LM perplexity ranking
```

candidates=${folder}/final_candidates.pkl
tgt_sent="TSAR-2022-Shared-Task/datasets/test/tsar2022_pt_test_none.tsv"
folder=${folder}
model=google/mt5-large
CUDA_VISIBLE_DEVICES=0 python prob_tgt.py -candidates ${candidates} -folder ${folder} -model ${model} -tgt_sent ${tgt_sent}
```

## Produce the final ranking
```
folder=bert-large-portuguese-cased_0_MASK_kmeans4_vec_NEW_nofilter_tsar2022_pt_tgtwords_weight_mt5-large_bs20_nolenpena_best
folder=test_fixed_pt
emb=${folder}/neuralmind_bert-large-portuguese-cased_candidates2reranking_score.pkl
python final_rank.py -lang pt -emb ${emb} -folder ${folder} -w_emb 3 -w_t5 2  -w_freq 0 -w_ppl 1
```
