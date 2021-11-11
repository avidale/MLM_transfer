# the goal is to run inference on the file /home/dale/dialogue-censor/parallel/test_toxic_parallel.txt 
# everything runs on the server NLP1

# preparation
cd /home/dale/projects/MLM_transfer/
source mlm_env/bin/activate
export CUDA_VISIBLE_DEVICES=6

# FT_CLS_ATTENTION_BASED
# data prerocessing
cp /home/dale/dialogue-censor/parallel/test_toxic_parallel.txt raw_data/toxic/sentiment.infer.1 
cp configs/cbert_toxic_attention_based.config run.config
python preprocess_attention_based.py raw_data/toxic/sentiment.infer.1 label toxic.infer.1 toxic processed_data_attention_based
cp processed_data_attention_based/toxic/toxic.infer.1.data.label processed_data_attention_based/toxic/infer.data.label

# inference
cp configs/cbert_toxic_attention_based.config run.config
python transfer.py run.config bert_ft_cls infer
python convert_outputs.py evaluation/outputs/toxic/bert_ft_cls_attention_based
cp evaluation/outputs/toxic/bert_ft_cls_attention_based/sentiment.infer.1.cbert.txt /home/dale/dialogue-censor/parallel/outputs_parallel/mask_infill_bert_ft_cls_attention_based.txt


# FT_FUSION_METHOD
# preprocess frequency
cp configs/cbert_toxic_frequency_ratio.config run.config
python preprocess_train.py raw_data/toxic/sentiment.infer.1 toxic.train.1 label 30000 6 toxic.infer.1 toxic processed_data_frequency_ratio
cp processed_data_frequency_ratio/toxic/toxic.infer.1.data.label processed_data_frequency_ratio/toxic/infer.data.label

# preprocess fusion
cp configs/cbert_toxic_fusion_method.config run.config
python preprocess_fusion_method_for_test.py raw_data/toxic/sentiment.infer.1 toxic.train.1 label 40000 1 toxic.infer.1 toxic processed_data_fusion_method
cp processed_data_fusion_method/toxic/toxic.infer.1.data.label processed_data_fusion_method/toxic/infer.data.label

# inference
cp configs/cbert_toxic_fusion_method.config run.config
python transfer.py run.config bert_ft infer
python convert_outputs.py evaluation/outputs/toxic/bert_ft_fusion_method

cp evaluation/outputs/toxic/bert_ft_fusion_method/sentiment.infer.1.cbert.txt /home/dale/dialogue-censor/parallel/outputs_parallel/mask_infill_bert_ft_fusion_method.txt



