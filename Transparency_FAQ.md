# What does the repository contain?

This repository contains the NOTSOFAR-1 baseline system for speaker attributed distant meeting transcription. 
It includes a CSS (continuous speech separation model) model trained on NOTOSFAR's simulated training dataset, along with CSS inference, diarization, and speech recognition modules.
The repository also contains inference pipeline code, data downloading and processing code, and code to measure word-error rate-based metrics.

# What are these components’ intended use(s)?

The scientific community will use this system as a baseline to perform research on distant speaker attributed speech recognition. They will be able to use their own data and methods and extend this baseline to improve it. Then, the evaluation framework could be used to assess the level of improvement their models have achieved.

# What are the limitations of these components?

1.	The system components are not fine-tuned to be considered as state-of-the-art candidates for speech separation, diarization, and recognition. We only provide a baseline approach.
2.	The inference code we provide is compatible only with the original models and may not be compatible after any modifications is applied to the models.
3.	The inference code assumes a certain data structure as documented inside the code and may not work properly if one decides to utilize the inference code with data of a different structure. 
4.	The evaluation framework gives a limited depth of analysis on the performance of the model. There may be low correlation between the results of the evaluation metrics and the subjective assessment of the user on what they consider a good performance.
5.	The existence of environmental noises with high energy levels may cause performance degredation that has not been quantified in our experiments.
6.	The models we provide are intended to be used with English speakers, and may present different performance evaluation depending on the English accent and dialects. We have not evaluated it regarding other languages other than English.
