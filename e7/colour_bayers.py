import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sys
# import cv2


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 113, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 186, 186),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])
# print(name_to_rgb)

def rgb_to_lab_transform(X):
    X_rgb = X.reshape(-1, 1, 3)
    X_lab = rgb2lab(X_rgb).reshape(-1, 3)
    return X_lab

def plot_predictions(model, lum=70, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)


def main(infile):
    data = pd.read_csv(infile)
    X_rgb_org = data[['R', 'G', 'B']]
    # print(X_rgb_org)
    X_rgb = data[['R', 'G', 'B']].values / 255.0 # array with shape (n, 3). Divide by 255 so components are all 0-1.
    y = data['Label'].values # array with shape (n,) of colour words.

    X_train_rgb, X_test_rgb, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2,random_state=42)
    
    # print(X_rgb)
    # print(y)
    # TODO: build model_rgb to predict y from X.
    model_rgb = GaussianNB()
    model_rgb.fit(X_train_rgb, y_train)
    
    # TODO: print model_rgb's accuracy score
    # TODO: build model_lab to predict y from X by converting to LAB colour first.
    # TODO: print model_lab's accuracy score
    y_pred_rgb = model_rgb.predict(X_test_rgb)
    accuracy_rgb = accuracy_score(y_test, y_pred_rgb)
    print(f"Naïve Bayes Classifier (RGB) Accuracy: {accuracy_rgb:.4f}")
    
     # Convert RGB to LAB color space
     
    # print(X_rgb, "\n")
    # print(X_rgb.reshape(-1,1,3))
    # print("\n")
    # X_lab = rgb2lab((X_rgb.reshape(-1, 1, 3))).reshape(-1, 3)
    # X_lab[:, 0] /= 100  # Normalize L to [0,1]
    # X_lab[:, 1:] = (X_lab[:, 1:] + 128) / 255   # Normalize A, B to [0,1]
    # X_lab_df = pd.DataFrame(X_lab)
    # X_lab_df.to_csv("out_X_lab.csv")
    # print(X_lab)
    # print([X_lab[:, 1:].max])
    
    lab_pipeline = Pipeline([
        ('rgb_to_lab', FunctionTransformer(rgb_to_lab_transform, validate=False)),
        ('classifier', GaussianNB())
    ])
    # X_train_lab, X_test_lab, y_train_lab, y_test_lab = train_test_split(X_lab, y, test_size=0.4, random_state=42)
    # print(X_train_lab)
    # Train Naïve Bayes classifier on LAB data
    model_lab = lab_pipeline.fit(X_train_rgb, y_train)
    # model_lab = GaussianNB()
    # model_lab.fit(X_train_lab, y_train_lab)
    
    # Evaluate accuracy for LAB
    y_pred_lab = lab_pipeline.predict(X_test_rgb)
    accuracy_lab = accuracy_score(y_test, y_pred_lab)
    print(f"Naïve Bayes Classifier (LAB) Accuracy: {accuracy_lab:.4f}")
    
    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plt.show()
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1])
