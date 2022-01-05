from typing import List

import segmentation_models_pytorch as smp
import torch


default_params = {'lr': 0.0001, 'classes': 1, 'in_channels': 4,
                  'activation': 'sigmoid', 'loss': smp.utils.losses.DiceLoss(),
                  'device': 'cuda', 'epochs': 10}


def create_two_simple_networks(**params):
    """ Initialise two neural networks for image segmentation """

    return None


def create_three_simple_networks(**params):
    raise NotImplementedError()


def create_four_simple_networks(**params):
    raise NotImplementedError()


def create_two_advanced_networks(**params):
    raise NotImplementedError()


def _create_segmentation_net(params: dict):
    """ Create neural network with desired parameters """
    if params is None:
        params = {**params, **default_params}

    nn_model = smp.PAN(encoder_name=params['encoder_name'],
                       encoder_weights=params['encoder_weights'],
                       in_channels=params['in_channels'],
                       classes=params['classes'],
                       activation=params['activation'])
    optimizer = torch.optim.Adam(params=nn_model.parameters(), lr=params['lr'])
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    train_epoch = smp.utils.train.TrainEpoch(nn_model, loss=params['loss'],
                                             metrics=metrics,
                                             optimizer=optimizer,
                                             device=params['device'],
                                             verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(nn_model, loss=params['loss'],
                                             metrics=metrics,
                                             device=params['device'],
                                             verbose=True)

    return train_epoch, valid_epoch