import numpy as np
from .base import Module
from scipy import special


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return grad_output * (input > 0)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return 1.0 / (1 + np.exp(-1 * input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return grad_output * np.exp(-1 * input) / np.power((np.exp(-1 * input) + 1), 2)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return special.softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # Подсмотрено тут: 
        # https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-activation-function-and-its-derivative-jacobian-in-python/
        softmax = special.softmax(input, axis=1)
        tensor1 = np.einsum('ij,ik->ijk', softmax, softmax)
        tensor2 = np.einsum('ij,jk->ijk', softmax, np.eye(input.shape[1]))
        dSoftmax = tensor2 - tensor1
        return np.einsum('ijk,ik->ij', dSoftmax, grad_output)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # Подсмотрено тут: 
        # https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-activation-function-and-its-derivative-jacobian-in-python/
        softmax = special.softmax(input, axis=1)
        tensor1 = np.einsum('ij,ik->ijk', softmax, softmax)
        tensor2 = np.einsum('ij,jk->ijk', softmax, np.eye(input.shape[1]))
        dSoftmax = tensor2 - tensor1
        return np.einsum('ijk,ik->ij', dSoftmax, grad_output / softmax)