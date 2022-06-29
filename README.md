### Baseline
```bash
# Vanilla ERM for Long-tailed CIFAR10 with label noise
python main.py --cfg ./config/ImbalanceCifar10/feat_uniform.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5

# Vanilla ERM for Long-tailed CIFAR100 with label noise
python main.py --cfg ./config/ImbalanceCifar100/feat_uniform.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5

# Vanilla ERM-DRW for Long-tailed CIFAR10 with label noise
python main.py --cfg ./config/ImbalanceCifar10/feat_uniformdrw.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5

# Vanilla ERM-DRW for Long-tailed CIFAR100 with label noise
python main.py --cfg ./config/ImbalanceCifar100/feat_uniformdrw.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5
```
### Robust Long-Tailed Learning under Label Noise (RoLT)

```bash
# RoLT for Long-tailed CIFAR10 with label noise
python main.py --cfg ./config/ImbalanceCifar10/feat_uniform.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5 --cleaning

# RoLT-DRW for Long-tailed CIFAR10 with label noise
python main.py --cfg ./config/ImbalanceCifar10/feat_uniformdrw.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5 --cleaning

# RoLT for Long-tailed CIFAR100 with label noise
python main.py --cfg ./config/ImbalanceCifar100/feat_uniform.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5 --cleaning

# RoLT-DRW for Long-tailed CIFAR100 with label noise
python main.py --cfg ./config/ImbalanceCifar100/feat_uniformdrw.yaml --imb_type exp --imb_factor 0.01 --noise_mode imb --noise_ratio 0.5 --cleaning
```
