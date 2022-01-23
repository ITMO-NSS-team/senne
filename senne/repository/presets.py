from copy import deepcopy
from typing import List

import segmentation_models_pytorch as smp
import torch


default_params = {'network': smp.Unet, 'lr': 0.0001, 'classes': 1, 'in_channels': 4,
                  'activation': 'sigmoid', 'loss': smp.utils.losses.DiceLoss(eps=1.),
                  'device': 'cuda', 'epochs': 10, 'batch_size': 3,
                  'encoder_weights': 'imagenet', 'encoder_name': 'resnet18'}


def create_two_simple_networks(**params) -> List[dict]:
    """ Initialise parameters for two neural networks for image segmentation """
    first_params = {'network': smp.Unet,
                    'lr': 0.0001,
                    'loss': smp.utils.losses.DiceLoss(),
                    'epochs': 20,
                    'encoder_name': 'resnet18',
                    'encoder_weights': 'imagenet',
                    'activation': 'sigmoid'}
    first_params = _update_parameters(first_params)

    second_params = {'network': smp.PAN,
                     'lr': 0.0001,
                     'loss': smp.utils.losses.JaccardLoss(),
                     'epochs': 20,
                     'encoder_name': 'resnet18',
                     'encoder_weights': 'swsl',
                     'activation': 'sigmoid'}
    second_params = _update_parameters(second_params)
    return [first_params, second_params]


def create_three_simple_networks(**params) -> List[dict]:
    raise NotImplementedError()


def create_four_simple_networks(**params) -> List[dict]:
    raise NotImplementedError()


def create_two_advanced_networks(**params) -> List[dict]:
    raise NotImplementedError()


def segmentation_net_builder(params: dict):
    """ Create neural network with desired parameters """
    nn_model = params['network'](encoder_name=params['encoder_name'],
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

    return train_epoch, valid_epoch, nn_model


def _update_parameters(custom_dictionary: dict):
    obtained_parameters = deepcopy(default_params)
    for parameter in list(obtained_parameters.keys()):
        if custom_dictionary.get(parameter) is not None:
            obtained_parameters[parameter] = custom_dictionary.get(parameter)

    return obtained_parameters
