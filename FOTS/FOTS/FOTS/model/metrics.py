from torchmetrics import ConfusionMatrix # Confusion Matrix
import torch # Diagonal Summations
import matplotlib.pyplot as plt # Plotting Score map

EPS = 1e-10

class Metrics():
    def __init__(self):
        self.confusion = ConfusionMatrix(num_classes=2)

    def get_metrics(self, predictedgrad, theoreticalgrad):

        predicted = predictedgrad.cpu().detach()
        theoretical = theoreticalgrad.cpu().detach()

        confusion = False

        for i in range(predicted.size()[1]):
            plt.imshow(predicted[i].permute(1, 2, 0))
            plt.imshow(theoretical[i].permute(1, 2, 0))
            plt.show()  

            imconfusion = self.confusion(predicted[i],theoretical[i])
            print("IMCONFS",imconfusion)
            if confusion:
                confusion += imconfusion
            else:
                confusion = imconfusion


        # Flip classes, Torch makes the positive class negative (and viceversa) for some reason 
        confusion = confusion.flip(0)
        confusion = confusion.flip(1)

        accuracy = get_accuracy(confusion)
        dice = dice_coefficient(confusion)
        pixel_acc = per_class_pixel_accuracy(confusion)

        print("Accuracy:", accuracy, "\nConfusion", confusion,"\nDiced",  dice, "\bPixel Accuracy", pixel_acc)

        return accuracy, confusion, dice , pixel_acc

def get_accuracy(confusion):
    correct = torch.diag(confusion)
    correct = correct.sum()
    total = confusion.sum()
    accuracy = correct/total

    return accuracy

def dice_coefficient(confusion):
    # Computes the Sorensen–Dice coefficient or F1 score.
    # Returns the average per-class dice coefficient.

    # Get array with all correct predictions per class 
    A_inter_B = confusion[0][0]

    # Get Arrays with the sum along each dimension of the matrix
    A = confusion.sum(dim=1)
    B = confusion.sum(dim=0)

    # Get dice coefficient
    dice = (2 * A_inter_B) / (A + B + EPS)  # A + B is an array
    # with the total predictions related to each class 

    # Compute the average
    avg_dice = nanmean(dice)

    return avg_dice


def per_class_pixel_accuracy(confusion):
    
    # Get array with all correct predictions per class
    correct_per_class = torch.diag(confusion)

    # Get array containing the total ammount of occurences per class
    total_per_class = confusion.sum(dim=1)

    # Compute array with the accuracy per class
    per_class_acc = correct_per_class / (total_per_class + EPS)

    # Get the mean accuracy per class
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def nanmean(x):
    # Compute the arithmetic mean ignoring any NaNs.
    return torch.mean(x[x == x])