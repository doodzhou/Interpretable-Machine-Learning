(1) Peng Zhou, Qihui Tong, Shiji Chen, Yunyun Zhang, Xindong Wu. EACE: Explain Anomaly via Counterfactual Explanations, Pattern Recognition, 164:111532, 2025.

@article{ZHOU2025111532,
title = {EACE: Explain Anomaly via Counterfactual Explanations},
journal = {Pattern Recognition},
volume = {164},
pages = {111532},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111532},
url = {https://www.sciencedirect.com/science/article/pii/S003132032500192X},
author = {Peng Zhou and Qihui Tong and Shiji Chen and Yunyun Zhang and Xindong Wu},
keywords = {Interpretable machine learning, Counterfactual explanation, Anomaly detection, Genetic algorithm},
abstract = {Anomaly detection aims to identify data points that deviate from the prevailing data distribution. Despite numerous anomaly detection models, there is a prevailing oversight in their interpretability, specifically regarding the rationale behind classifying a specific data point as an anomaly. Therefore, Interpretable Machine Learning has become a current research hotspot and is crucial for users to trust models. As one of the representative models, Counterfactual Explanation (CFE) methods generate alternative scenarios different from the observed data to explain model decisions. CFE tries to answer how the model’s output would change if certain factors (features) were altered. However, most existing CFE methods are designed for classification tasks, and it is a challenge for them to transform anomalies into counterfactual explanation samples effectively. To overcome this limitation, we propose a novel method for Explaining Anomaly via Counterfactual Explanation named EACE. Specifically, based on existing CFE methods’ limitations in handling anomalies, we propose a novel optimization objective by incorporating density loss and boundary loss. Meanwhile, we improved the genetic algorithm to solve this optimization problem since the new loss function is not differentiable. To evaluate the quality of the generated counterfactual explanations, we compare comprehensively with state-of-the-art counterfactual explanation methods and feature importance-based explanation methods. Experimental results demonstrate that EACE has a notable ability to convert anomalies into counterfactual explanation samples that are highly aligned with the normal cluster.}
}


(2) Zhang, Yu and Zhang, Yunyun and Sun, Xiuwen and Zhou, Peng, Plausible and Robust Counterfactual Explanation via Local Distribution Consistency, Neurocomputing, 702:134643, 2026 

@article{ZHANG2026134643,
title = {Plausible and robust counterfactual explanation via local distribution consistency},
journal = {Neurocomputing},
volume = {702},
pages = {134643},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.134643},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226020412},
author = {Yu Zhang and Yunyun Zhang and Xiuwen Sun and Peng Zhou},
keywords = {Explainable machine learning, Counterfactual explanations, Local distribution consistency, Particle swarm optimization},
abstract = {In explainable machine learning, counterfactual explanations reveal key factors influencing model decisions by constructing hypothetical scenarios (for example, “if the inputs changed, the outputs would differ.”). Most existing methods generate counterfactuals by minimizing the distance between the generated instance and the original input, as well as other constraints. However, prioritizing proximity can degrade plausibility and robustness. Our analysis attributes this trade-off to neglecting consistency with the target class’s local data distribution during counterfactual generation. To address this, we propose a new method that balances proximity, plausibility, and robustness by enforcing local distribution consistency. Specifically, we introduce a novel loss function consisting of two terms: Local Density Loss and Feature Deviation Loss. The Local Density Loss steers generated counterfactuals toward high-density regions of the target class, improving plausibility. The Feature Deviation Loss preserves proximity by restricting modifications to salient features, encouraging alignment with the target class in those dimensions, while minimizing changes to less important features to enhance robustness. Additionally, we propose an enhanced particle swarm optimization (PSO) algorithm that integrates feature weights into the velocity update to accelerate the targeted search. Extensive experiments and visualized case studies illustrate that our new method can effectively balance plausibility, robustness, and proximity.}
}


(3) Peng Zhou, Zhiyong Huang, Yuanting Yan. Detection-Explanation-Improvement: A Closed-Loop Framework of Enhancing Anomaly Detection with Counterfactual Explanations, IJCAI 2026.

(4) Zhou, Peng, Shiji Chen, Yunyun Zhang, Lin Mu, and Xindong Wu. "Multi-Class Counterfactual Explanations via Direction and Aggregation Loss." Available at SSRN 6328216.
