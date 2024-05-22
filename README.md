# Adaptive Instrument Design for Indirect Experiments

Yash Chandak, Shiv Shankar, Vasilis Syrgkanis, Emma Brunskill.

Twelfth International Conference on Learning Representations (ICLR 2024) 


https://arxiv.org/abs/2312.02438


## Code

Install requirements

```shell
pip install -r requirements.txt
```

To set the algorithm and the domain use the following arguments

```shell
--algo_name # Options {uniform, oracle, proposed}
--env_name # Options {synthetic, TripAdvisor}
--n_IV # Number of instruments
```

Other arguments and hyper-parameters can be found in the respective ``run_{TA/CIV/IV}.py`` files.

Some examples:

1. To run the proposed method on TripAdvisor:
```shell
run_TA.py --algo_name proposed --env_name semi-synthetic
``` 

2. To run synthetic conditional IV with 40 instruments
```shell
run_CIV.py --n_IV 40 --algo_name proposed --env_name synthetic
```
3. To run synthetic IV with 10 instruments 

```shell
run_IV.py --n_IV 10 --algo_name proposed --env_name synthetic
```

# Citation

```bib
@inproceedings{chandak2023adaptive,
  title={Adaptive Instrument Design for Indirect Experiments},
  author={Chandak, Yash and Shankar, Shiv and Syrgkanis, Vasilis and Brunskill, Emma},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```