import matplotlib.pyplot as plt
import numpy as np

def display_examples(examples, lables):
    plt.figure(figsize=(5, 5))
    for i in range(20):
        indx = np.random.randint(0, examples.shape[0]-1)
        img = examples[indx]
        lable = lables[indx]

        plt.subplot(4, 5, i+1)
        plt.title(str(lable))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()
