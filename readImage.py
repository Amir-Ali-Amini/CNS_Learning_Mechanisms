import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def readImage(
    image_address,
    n=10,
    prefix="./amini-amirali-610399102-cns-p03/code/images/",
    show_image=True,
    flatten=True,
):
    # Read the grayscale image
    try:
        image = cv2.imread(prefix + image_address, cv2.IMREAD_GRAYSCALE)
        # Resize the image to a square size n*n
        resized_image = cv2.resize(image, (n, n))

        # Display the resized image
        if show_image:
            # cv2.imshow("Resized Image", resized_image)
            plt.imshow(resized_image, cmap="gray")
            plt.axis("off")  # Remove axis
            plt.show()

        # Return the pixel matrix of the resized image
        return list(np.array(resized_image).reshape(-1)) if flatten else resized_image
    except Exception as e:
        raise ValueError(
            f"{os.getcwd()}\nNo such file {prefix + image_address}\n\n\n\n{e}"
        )
