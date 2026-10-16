# AI301 Deep Learning — Assignment #2: Convolutional Neural Networks

## EDA and Metric Analysis

- [ ] Perform Exploratory Data Analysis (EDA)
- [ ] Analyze the proposed target metric: strengths, weaknesses, potential edge cases
- [ ] Suggest complementary metrics to capture other aspects of performance
- [ ] Propose an alternative target metric, with justification

## Validation

- [ ] Design a validation strategy based on common sense and expected correlation with Public leaderboard scores
- [ ] Explain the motivation for the chosen strategy
- [ ] Discuss possible reasons for low or high correlation with leaderboard results
- [ ] Perform adversarial validation if necessary (with the same model as the main task)

## CNN Model

- [ ] Set up experiment and logging infrastructure using high-level libraries (like PyTorch Lightning and W&B)
- [ ] Use an existing architecture ([pytorch-image-models](https://github.com/huggingface/pytorch-image-models) CNN zoo)
- [ ] Experiment with different optimizers, schedulers, hyperparameters and all other elements covered in the lecture materials
- [ ] Implement your own CNN OR add your own building block to an existing architecture; compare results with the previous CNN
- [ ] Collect all relevant results:
  - [ ] Loss/metric curves
  - [ ] Gradient flow visualizations
  - [ ] GradCAM visualization
  - [ ] Parameter tuning logs, etc.
- [ ] Submit predictions and compare Public leaderboard metrics with local validation results
- [ ] Conclude whether the model is adequate

## Ensembling

- [ ] Combine several CNN models using ensembling (NOT stacking)
- [ ] [Optional] Use Logistic Regression to perform stacking
- [ ] Submit predictions to obtain Public leaderboard metrics; compare with local validation results

## Final Report (LaTeX or `README.md`)

- [ ] Data insights
- [ ] Feature engineering
- [ ] Tried models
- [ ] Chosen hyperparameters
- [ ] Metrics and optimization curves
- [ ] Model insights and conclusions

## Notes

- [ ] No additional or external data at any step (training, validation, EDA, etc.); violation means instant BAN. Pretrained models allowed only on ImageNet ([pytorch-image-models](https://github.com/huggingface/pytorch-image-models) or other libraries)
- [ ] Include failed ideas and experiments in the report, with what they revealed about the data, task, or algorithms
- [ ] Keep EDA concise: only key aspects (missing values, target distribution); drop parts without meaningful conclusions; no verbose Kaggle-style EDA
- [ ] Prepare the presentation carefully: no messy speech, not too much code or too many files; decide in advance what to highlight
- [ ] No Keras, TensorFlow, or Chainer; violation means instant BAN
- [ ] Analyze the train/test split; consider time and group factors in validation
- [ ] Use the same splits for all ML/DL approaches so results are comparable
- [ ] Huge CV vs Kaggle LB mismatch: consider adversarial validation
- [ ] Final report includes both CV and LB scores for all experiments
