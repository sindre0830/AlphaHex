import math
import tqdm
import torch.utils.data

# Cantor encoding
# https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
def cantor_encode(x, y):
    if x < 0 or y < 0:
        raise ValueError(f"{x} and {y} cannot be paired due to negative values")
    z = int(0.5 * (x + y) * (x + y + 1) + y)
    if (x, y) != cantor_decode(z):
        raise ValueError(f"{x} and {y} cannot be paired due to large number")
    return z


# Cantor decoding
# https://en.wikipedia.org/wiki/Pairing_function#Inverting_the_Cantor_pairing_function
def cantor_decode(z):
    w = math.floor((math.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    return x, y

# Set prefix in progressbar and update output.
def setProgressbarPrefix(
    progressbar: tqdm.tqdm,
    trainLoss: float = 0.,
    trainAccuracy: float = 0.,
    valLoss: float = 0.,
    valAccuracy: float = 0.,
    modelSaved: bool = ''
):
    trainLossStr = f'Train loss: {trainLoss:.4f}, '
    trainAccuracyStr = f'Train acc: {trainAccuracy:.4f}, '
    valLossStr = f'Val loss: {valLoss:.4f}, '
    valAccuracyStr = f'Val acc: {valAccuracy:.4f}, '
    modelSaved = f'Saved: {modelSaved!s:>5}'
    progressbar.set_postfix_str(trainLossStr + trainAccuracyStr + valLossStr + valAccuracyStr + modelSaved)


# Generates progressbar for iterable used in model training.
def getProgressbar(iter: torch.utils.data.DataLoader, epoch, epochs):
    width = len(str(epochs))
    progressbar = tqdm.tqdm(
        iterable=iter,
        desc=f'Epoch {(epoch + 1):>{width}}/{epochs}',
        ascii='░▒',
        unit=' steps',
        colour='blue'
    )
    setProgressbarPrefix(progressbar)
    return progressbar
