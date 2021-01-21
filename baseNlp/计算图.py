import torch
from torch.utils.data import DataLoader
if __name__ == '__main__':
    x = torch.tensor([1.])
    w = torch.tensor([2.], requires_grad=True)  # model.train()
    b = torch.tensor([0.], requires_grad=True)

    y = torch.addcmul(b, w, x)

    y.backward()
    print(w.grad)
    print(b.grad)
    print(x.grad)

    train_loder = DataLoader()


