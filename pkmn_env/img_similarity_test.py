import os

import cv2
import hnswlib
import numpy as np

def test():

    _, _, images = next(os.walk("image_tests"))

    tested = sorted([image for image in images if "observed" in image])

    loaded_images = [cv2.imread("image_tests/" + image, cv2.IMREAD_GRAYSCALE) for image in tested]


    knn_index = hnswlib.Index(space='l2', dim=72*72)
    knn_index.init_index(
        max_elements=1000, ef_construction=100, M=16
    )

    # 45_239_656

    # 27_000_000

    for i, (image, name) in enumerate(zip(loaded_images, tested)):
        frame_vector = np.float32(image.flatten())/255.
        if i == 0:
            knn_index.add_items(
                frame_vector, np.array([i])
            )
        else:
            labels, distances = knn_index.knn_query(frame_vector, k=1)
            distance = distances[0][0]
            print(distance, name)
            #if distance > 27_000_000:
            #print("r!")
            knn_index.add_items(
                frame_vector, np.array([i])
            )

if __name__ == '__main__':
    test()