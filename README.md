The PixelRL Denoiser for colored images
----------
Class project for EEE 598 (Reninforcement Learning in Robotics) 

- Action sets are written in State.py
- FCN extended to accept color images

# Training
- To train, open a terimnal in the project folder and run
```
python Train_torch.py
```

# Testing
- To run inference, open a terimnal in the project folder and run
```
python Tst.py
```


# Demo
- Website to test code: [link](https://mnthnx64.github.io/image-transceiver/)



# Requirements and Dependencies
- python 3.5+
- python dependencies are in requirements.txt



# Citation

Please quote the original paper [PixelRL](https://arxiv.org/abs/1811.04323).

```
@inproceedings{aaai_furuta_2019,
    author={Ryosuke Furuta and Naoto Inoue and Toshihiko Yamasaki},
    title={Fully Convolutional Network with Multi-Step Reinforcement Learning for Image Processing},
    booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
    year={2019}
}
@article{furuta2020pixelrl,
    title={PixelRL: Fully Convolutional Network with Reinforcement Learning for Image Processing},
    author={Ryosuke Furuta and Naoto Inoue and Toshihiko Yamasaki},
    journal={IEEE Transactions on Multimedia (TMM)},
    year={2020},
    volume={22},
    number={7},
    pages={1704-1719}
}
```
