import numpy
from torchvision import datasets, transforms

def compute_mean_std(dataset):
    """compute the mean and std of cifar10 dataset
    Args:
    cifar10_training_dataset or cifar10_test_dataset
    witch derived from class torch.utils.data

    Returns:
    a tuple contains mean, std value of entire dataset
    """

    mean_r = numpy.mean(dataset.data[:, :, :, 0]) / 255.0
    mean_g = numpy.mean(dataset.data[:, :, :, 1]) / 255.0
    mean_b = numpy.mean(dataset.data[:, :, :, 2]) / 255.0

    std_r = numpy.std(dataset.data[:, :, :, 0]) / 255.0
    std_g = numpy.std(dataset.data[:, :, :, 1]) / 255.0
    std_b = numpy.std(dataset.data[:, :, :, 2]) / 255.0

    mean = mean_r, mean_g, mean_b
    std = std_r, std_g, std_b

    return mean, std

if __name__ == '__main__':
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True)
    print(compute_mean_std(dataset_train))
    print(compute_mean_std(dataset_test))
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), transforms.Resize((224, 224))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    print(compute_mean_std(dataset_train))
    print(compute_mean_std(dataset_test))
    #print(sum([1,3,4,6,4]))
