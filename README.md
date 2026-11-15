# Offer Policy Optimization

Which offer should each bank customer get: cashback, an automatic savings offer, a reminder, or nothing? The goal is a policy that maximizes total reward in a customer simulator while keeping churn low. The project compares rule based policies built on customer segments with a policy learned by Q-learning.

Everything is in one notebook: [`offer_policy.ipynb`](offer_policy.ipynb).

## Setup of the problem

- 500 customers with five attributes: balance, average spending, salary, savings rate and responsiveness to offers.
- Four possible actions: `auto_save_offer`, `cashback`, `reminder`, `none`.
- A simulator (`CustomerEnvironment`) that returns the reward and tells whether the customer churned after each action.

The customer data and the simulator are confidential and are not included in this repository.

## What the notebook does

1. **EDA.** Distributions, outliers and correlations. Savings rate and responsiveness move together, which already suggests that responsive customers should get savings related offers.
2. **Segmentation.** K-means on standardized features. The elbow method and the silhouette score both point to 8 clusters. A 3 cluster version is kept for comparison, and both are visualized with PCA and t-SNE.
3. **Rule based policies.** One offer per cluster, chosen from the cluster profile, compared with a random policy in the simulator.
4. **Segment analysis.** Reward and churn per cluster and per offer, then a second version of the policies where some clusters switch offers after the first 25 steps.
5. **Reinforcement learning.** Tabular Q-learning with an epsilon-greedy policy. The state is a discretized customer profile, built in two ways (per feature clustering and quartiles). The bins are fitted on the train customers only, and the hyperparameters are tuned with cross-validation.

## Results

| Policy | Total reward | Churn rate |
| --- | ---: | ---: |
| Random | 91.3M | 5.0% |
| 3 clusters, first version | 128.9M | 5.6% |
| 8 clusters, first version | 132.0M | 5.4% |
| 8 clusters, reward optimized | 143.3M | 18.8% |
| 3 clusters, reward optimized | 135.0M | 29.0% |

The reward optimized cluster policies earn more, but churn roughly triples, so they are not a good trade.

The Q-learning policy was evaluated on a train / test split of customers (400 / 100):

| Q-learning state | Avg reward per customer (test) | Churn (test) |
| --- | ---: | ---: |
| Per feature clusters | 297.1k | 9.0% |
| Quartiles | 296.4k | 10.0% |

That is a higher reward per customer than any of the cluster policies (264k to 287k), with churn well below the reward optimized ones. The comparison is not perfectly like for like, because the cluster policies were scored on all 500 customers, and a fresh test set would be needed to rank them properly.

## Reproducibility

The customer data and the simulator are confidential and are not part of this repository, so the notebook cannot be re-run from here. All outputs are saved in the notebook, so the full analysis, tables and charts can be read directly on GitHub. `requirements.txt` lists the libraries that were used. The customer simulator is a compiled Python 3.9 module, so the original run used Python 3.9.
