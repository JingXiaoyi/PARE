# Capturing Popularity Trends: A Simplistic Non-Personalized Approach for Enhanced Item Recommendation
This is the implementation for our CIKM 2023 paper:
>Capturing Popularity Trends: A Simplistic Non-Personalized Approach for Enhanced Item Recommendation.

## Enviroment
Please follow `requirements.txt`

## Abstract
 Recommender systems have been gaining increasing research attention over the years. Most existing recommendation methods focus on capturing users' personalized preferences through historical user-item interactions, which may potentially violate user privacy. Additionally, these approaches often overlook the significance of the temporal fluctuation in item popularity that can sway users' decision-making. To bridge this gap, we propose ***P****opularity-***A***ware ****Re****commender* (*PARE*), which makes non-personalized recommendations by predicting the items that will attain the highest popularity. *PARE* consists of four modules, each focusing on a different aspect: popularity history, temporal impact, periodic impact, and side information. Finally, an attention layer is leveraged to fuse the outputs of four modules. To our knowledge, this is the first work to explicitly model item popularity in recommendation systems. Extensive experiments show that *PARE* performs on par or even better than sophisticated state-of-the-art recommendation methods. Since *PARE* prioritizes item popularity over personalized user preferences, it can enhance existing recommendation methods as a complementary component. Our experiments demonstrate that integrating *PARE* with existing recommendation methods significantly surpasses the performance of standalone models, highlighting *PARE*'s potential as a complement to existing recommendation methods. Furthermore, the simplicity of *PARE* makes it immensely practical for industrial applications and a valuable baseline for future research.

## Dataset
You can download the Amazon Dataset from <http://jmcauley.ucsd.edu/data/amazon/index_2014.html>

## Run *PARE*
For Amazon Home and Kitchen dataset:
```
python ./code/main.py --dataset=reviews_Home_and_Kitchen_5 --beta=0.9
```
For Amazon Video Games dataset:
```
python ./code/main.py --dataset=reviews_Home_and_Kitchen_5 --beta=0.7
```

## Integrate *PARE* with baselines
```
python ./code/integrate.py --pare_output=pare.out --baseline_output=baseline.out
```

## Citation
