# Samples
- [BERT AutoEncoder SHAP for browser history](#bert-autoencoder-shap-for-browser-history)

## BERT[^1] AutoEncoder SHAP[^2] for browser history
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers shap scikit-learn pandas numpy tldextract tqdm matplotlib
cd file2risks
python run_bert_from_takeout.py /path/to/takeout-yyyyMMddThhmmssZ-*.zip
```

## References
[^1]:[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)
[^2]:[Explaining Anomalies Detected by Autoencoders Using SHAP](https://arxiv.org/pdf/1903.02407)