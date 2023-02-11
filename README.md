# Is My Data Leaking
Check if you have duplicates samples across different datasets or splits with one simple class.
Matrix Hashing enables checking huge datasets that wouldn't fit in memory.

## Example

```
from torchvision.datasets import cifar
import numpy as np
from Imdl import Imdl


def main():
    data_train = cifar.CIFAR10(root='./data', train=True, download=True)
    data_test = cifar.CIFAR10(root='./data', train=False, download=True)
    transform = lambda x: np.array(x[0])  # we take the image and convert it to numpy array
    imdl = Imdl(data_train, data_test, transform=transform, progress_bar=True)
    print(imdl.find_duplicates(progress_bar=True))


if __name__ == '__main__':
    main()
```