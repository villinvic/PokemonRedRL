import os

import cv2
import annoy
import numpy as np
import imagehash
from PIL import Image

def test():

    _, _, images = next(os.walk("image_tests"))

    tested = sorted([image for image in images if "observed" in image])

    np.random.shuffle(tested)

    loaded_images = []
    tested_images = []

    images = dict()

    for i, image in enumerate(tested):
        img = cv2.imread("image_tests/" + image, cv2.IMREAD_GRAYSCALE)[:-24]
        images[i] = img
        shape = (img.shape[0]//1 - 12, img.shape[1]//1)

        #img = cv2.resize(img, (img.shape[1]//1, img.shape[0]//1), interpolation=cv2.INTER_NEAREST)#[:-12]
        img_whash = imagehash.whash(Image.fromarray(img))
        img_phash = imagehash.phash(Image.fromarray(img))
        img_dhash = imagehash.dhash(Image.fromarray(img))

        v = np.uint8(img.flatten())
        # v = np.concatenate([img_whash.hash.astype('int').flatten(),
        #                     #img_phash.hash.astype('int').flatten(),
        #                     #img_dhash.hash.astype('int').flatten()
        #                     ])
        #v = img.flatten()

        if i < 4:
            tested_images.append(v)
            knn_index = annoy.AnnoyIndex(len(v), "euclidean")
        else:
            loaded_images.append(v)
            knn_index.add_item(i-4, v)

    knn_index.build(32, 1)

    # 45_239_656

    # 27_000_000

    for i, (image) in enumerate(tested_images):

        labels, distances = knn_index.get_nns_by_vector(image, n=1, search_k=64 , include_distances=True)
        # nearest_image = np.array(knn_index.get_item_vector(labels[0]), dtype=np.uint8)
        nearest_image = images[labels[0]+4]
        print(distances[0])
        cv2.imshow("test", images[i])
        cv2.imshow("nearest", nearest_image)
        cv2.waitKey(0)
def test2():

    _, _, images = next(os.walk("image_tests"))

    tested = sorted([image for image in images if "original" in image])

    loaded_images = [cv2.imread("image_tests/" + image, cv2.IMREAD_GRAYSCALE) for image in tested]

    for l in loaded_images:
        grayscale_downsampled_screen = cv2.resize(
            l,
            tuple(reversed((36, 40))),
            interpolation=cv2.INTER_AREA,
        )[:, :, np.newaxis]

        cv2.imwrite("tmp.jpeg", grayscale_downsampled_screen)
        input()



if __name__ == '__main__':
    test()