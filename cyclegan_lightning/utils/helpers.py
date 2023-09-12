import os, sys, shutil
import numpy as np
import torch.nn as nn


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def verify_checkpoint_availability(args):
    checkpoint_path = os.path.join(args.checkpoint, args.name)
    if os.path.exists(checkpoint_path) and not args.resume and not args.test_only:
        print("Previous checkpoints for", args.name, "will be deleted.")
        if (
            not args.debug
            and query_yes_no("Do you want to delete this folder?", default="no")
            == False
        ):
            print("Then change the '--name' argument.")
            exit()
        else:
            shutil.rmtree(checkpoint_path)

    return checkpoint_path


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_mosaic(real_batch, fake_batch, rec_batch):

    real_batch = np.transpose(real_batch, (0, 2, 3, 1))
    fake_batch = np.transpose(fake_batch, (0, 2, 3, 1))
    rec_batch = np.transpose(rec_batch, (0, 2, 3, 1))

    if real_batch.shape[-1] == 3:
        real_batch *= 255.0
        fake_batch *= 255.0
        rec_batch *= 255.0

    real_stack = np.vstack([real_batch[i] for i in range(real_batch.shape[0])])
    fake_stack = np.vstack([fake_batch[i] for i in range(fake_batch.shape[0])])
    rec_stack = np.vstack([rec_batch[i] for i in range(rec_batch.shape[0])])

    mosaic = np.hstack((real_stack, fake_stack, rec_stack))

    return mosaic


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
