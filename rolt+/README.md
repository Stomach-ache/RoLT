### Download CIFAR dataset

```bash
cd cifar
bash download_data.sh cifar10
bash download_data.sh cifar100
```

### Run

```bash
# DivideMix
python Train_cifar.py --dataset cifar10  --imb_factor 0.01 --noise_ratio 0.5 -b loss

# class-independent loss
python Train_cifar.py --dataset cifar10  --imb_factor 0.01 --noise_ratio 0.5 --cls_ind -b loss

# RoLT+ (class-independent distance)
python Train_cifar.py --dataset cifar10  --imb_factor 0.01 --noise_ratio 0.5 --cls_ind -b dist

# RoLT+ | warm_up 100 | drw 200
python Train_cifar.py --dataset cifar10  --imb_factor 0.01 --noise_ratio 0.5 --cls_ind -b dist -w 100 -d 200
```

