# What does the repository contain?

It contains 3 deep learning models that are open-sourced and are intended to perform speech separation, diarization, and recognition on both multi-channel and single-channel configurations. The speech separation and diarization models have been pre-trained with Microsoft data. The repository also contains an inference code, and an evaluation code that features word-error rate-based metrics.

# What are these components’ intended use(s)?

The scientific community will use these pre-trained models as a baseline to jump start a new cutting-edge technology. They will be able to use their own data and methods and add more deep architectures to this baseline to improve it. Then, the evaluation framework could be used to assess the level of improvement their models have achieved.

# What are the limitations of these components?

1.	They are not fine-tuned to be considered as state-of-the-art candidates for speech separation, diarization, and recognition. To achieve that, one would have to integrate additional training with sophisticated modeling. We only provide a baseline approach.
2.	The inference code we provide is compatible only with the original models and may not be compatible after any modifications is applied to the models.
3.	The inference code assumes a certain data structure as documented inside the code and may not work properly if one decides to utilize the inference code with data of a different structure. 
4.	The evaluation framework gives a limited depth of analysis on the performance of the model. There may be low correlation between the results of the evaluation metrics and the subjective assessment of the user on what they consider a good performance.
5.	The existence of environmental noises with high energy levels may cause performance degredation that has not been quantified in our experiments.
6.	The models we provide are intended to be used with English speakers, and may present different performance evaluation depending on the English accent and dialects. We have not conducted research regarding other languages other than English.
