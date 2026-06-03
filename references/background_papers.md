# Background Papers

## Primary implementation reference

1. Joseph Paul Cohen, Joseph D. Viviano, Paul Bertin, Paul Morrison, Parsa Torabian, Matteo Guarrera, Matthew P. Lungren, Akshay Chaudhari, Rupert Brooks, Mohammad Hashir, Hadrien Bertrand. "TorchXRayVision: A library of chest X-ray datasets and models." Medical Imaging with Deep Learning, 2022.
   - URL: https://arxiv.org/abs/2111.00595
   - Code: https://github.com/mlmed/torchxrayvision
   - Relevance: Provides the open-source CXR model library, pretrained classifiers, preprocessing pipeline, and pathology label set used by `cxr_classification_tool`.

## Baseline CXR classification reference

2. Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, Matthew P. Lungren, Andrew Y. Ng. "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." arXiv, 2017.
   - URL: https://arxiv.org/abs/1711.05225
   - Relevance: Representative DenseNet-style CXR disease classification baseline and motivation for pneumonia/pathology probability outputs.

## Dataset/model context

3. Jeremy Irvin et al. "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison." AAAI, 2019.
   - URL: https://arxiv.org/abs/1901.07031
   - Relevance: Provides context for large-scale CXR labels and uncertainty-aware chest radiograph classification.

4. Joseph Paul Cohen, Mohammad Hashir, Rupert Brooks, Hadrien Bertrand. "On the limits of cross-domain generalization in automated X-ray prediction." Medical Imaging with Deep Learning, 2020.
   - URL: https://arxiv.org/abs/2002.02497
   - Relevance: Important limitation reference: CXR classifiers can degrade across datasets/sites, so the tool must surface uncertainty and avoid final diagnostic claims.

## CXR report classification references

5. Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Y. Ng, Matthew P. Lungren. "CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT." EMNLP, 2020.
   - URL: https://arxiv.org/abs/2004.09167
   - Code: https://github.com/stanfordmlgroup/CheXbert
   - Relevance: Provides the open-source chest radiology report labeler and the CheXpert-compatible observation-label framing used by `cxr_report_labeling_tool`.

6. Stanford ML Group. "CheXpert labeler: NLP tool to extract observations from radiology reports."
   - Code: https://github.com/stanfordmlgroup/chexpert-labeler
   - Relevance: Provides the rule-based report-labeling baseline and label convention that motivated the lightweight local report labeler backend.
