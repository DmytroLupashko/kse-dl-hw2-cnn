# Cats and Dogs++

## Overview

Welcome to Cats and Dogs++, a multilabel image classification challenge where your task is to teach a model to recognize not only generic cats and dogs — but also their unique identities!

The dataset includes images of:

- Cats 🐱
- Dogs 🐶
- Moris and Motya — two distinctive cats
- Biatrix — a friendly dog
- Plus, some background images containing none of the above

Each image can contain one, several, or none of these classes. Your model should predict a binary label for each of the five categories (`Cat`, `Dog`, `Moris`, `Motya`, `Biatrix`), resulting in a multilabel target vector.

## Evaluation

Submissions are evaluated using the **Macro F1-score**, a balanced metric that equally weights the F1 of each class, regardless of its frequency in the dataset. This encourages models to perform well across all classes — even the less frequent ones like Moris, Motya, and Biatrix.

For each test image, participants must predict **five binary probabilities** corresponding to the presence of each class: `Cat`, `Dog`, `Moris`, `Motya`, and `Biatrix`.

The final score is computed as:

$$
\text{Macro F1} = \frac{1}{N} \sum_{i=1}^{N} F1_i
$$

## Submission File

For each image in the test set, you must predict whether each class is **present (1)** or **absent (0)**. Your submission file must contain a header and follow this format:

```csv
image_id,Cat,Dog,Moris,Motya,Biatrix
7e9de2837ab0f90176c0e6c2e30c2c136f1f00d9,1,1,1,1,1
11e8173dd91e7a00f8bdbc64067d6f167c9bbed2,1,1,1,1,1
415c62c8d85c0a13300ba25f4a5e6582eb468c31,1,1,1,1,1
4fbdedc390d83a9b264dc8e0ea803c4307a7f891,1,1,1,1,1
fb43e55f67815f61d3f298b8cc1947f278676ee5,1,1,1,1,1
```
