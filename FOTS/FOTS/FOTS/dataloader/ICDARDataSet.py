import typing  # Defining types
import pathlib  # Paths
import cv2  # Reading images
import numpy as np  # Handling images
from torch.utils.data import Dataset  # Pytorch Dataset class
import imgaug.augmentables.segmaps as ia_segmaps  # Score map handling
import torch  # Tensor creation
from .Transform import Transform  # Transform type
from ..utils.util import str_label_converter  # Convert a string into a label
from .datautils import (
    check_and_validate_polys,
    normalize_image,
)  # Validating polygons and normalizing image
from . import utils as data_utils  # Get score map geometry
from imgaug.augmentables.polys import (
    Polygon,
    PolygonsOnImage,
)  # Bounding boxes handling


class ICDARDataset(Dataset):
    def __init__(
        self,
        data_root,
        transform: Transform = None,
        scale: float = 0.25,
        size: int = 640,
        vis: bool = False,
        training: bool = True,
    ):
        data_root = pathlib.Path(data_root)

        self.images_root = data_root / "imgs"
        self.gt_root = data_root / "gt"
        self.training = training
        self.transform = transform
        self.vis = vis
        self.scale = scale
        self.size = size

        self.images, self.bboxs, self.transcripts = self.__loadGT()

    def __loadGT(self):
        all_bboxs = []
        all_texts = []
        all_images = []

        for image in self.images_root.glob("*.jpg"):

            gt = (
                self.gt_root
                / image.with_name("gt_{}".format(image.stem)).with_suffix(".txt").name
            )

            # Plot input image

            # cv2.waitKey(0)

            # pic = cv2.imread(imagep)

            # print(pic)

            # print(image)

            # cv2.imshow("pic",pic)

            with gt.open(mode="r") as f:
                bboxes = []
                texts = []
                for line in f:
                    text = line.strip("\ufeff").strip("\xef\xbb\xbf").strip().split(",")
                    #print("Transcript text: ", text)
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                    bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    transcript = text[8]
                    if transcript == "###" and self.training:
                        continue
                    bboxes.append(bbox)
                    texts.append(transcript)

                if len(bboxes) > 0:
                    bboxes = np.array(bboxes)
                    all_bboxs.append(bboxes)
                    all_texts.append(texts)
                    all_images.append(image)

        return all_images, all_bboxs, all_texts

    def visualize(
        self,
        image_name: str,
        image: np.ndarray,
        polygons: typing.List[typing.List[float]],
        score_map: np.ndarray,
        training_mask: np.ndarray,
    ):

        polygon_list = []
        for polygon in polygons:
            polygon_list.append(Polygon(np.array(polygon).reshape(4, 2)))

        polygons_on_image = PolygonsOnImage(polygons=polygon_list, shape=image.shape)
        new_image = polygons_on_image.draw_on_image(image)
        cv2.imwrite(image_name + ".jpg", new_image)

        score_map = ia_segmaps.SegmentationMapsOnImage(
            score_map.astype(dtype=np.uint8), shape=image.shape
        )
        new_image = score_map.draw_on_image(image.astype(dtype=np.uint8))
        cv2.imwrite(image_name + "_score.jpg", new_image[0])

        training_mask = ia_segmaps.SegmentationMapsOnImage(
            training_mask.astype(dtype=np.uint8), shape=image.shape
        )
        # new_image = training_mask.draw_on_image(image.astype(dtype=np.uint8))
        # cv2.imwrite(image_name + "_mask.jpg", new_image[0])

    def __getitem__(self, index):
        try:
            image_path = self.images[index]
            word_b_boxes = self.bboxs[index]  # num_words * 8
            transcripts = self.transcripts[index]

            im = cv2.imread((image_path).as_posix())

            # Plot input image
            # cv2.waitKey(0)
            # cv2.imshow("pic",im)

            image_path = pathlib.Path(image_path)

            num_of_words = word_b_boxes.shape[0]
            text_polys = word_b_boxes
            transcripts = [word for line in transcripts for word in line.split()]

            if num_of_words == len(transcripts):
                h, w, _ = im.shape
                # print("Image Shape: ", im.shape)
                text_polys = check_and_validate_polys(text_polys, (h, w))

                max_tries = 10
                if self.transform:
                    while True and (max_tries != 0):
                        transformed_im, transformed_text_polys = self.transform(
                            im, text_polys
                        )
                        valid_text_polys = [
                            polygon
                            for polygon in transformed_text_polys
                            if polygon.is_partly_within_image(image=im)
                        ]
                        if len(valid_text_polys) > 0:
                            text_polys = valid_text_polys
                            transcripts = [
                                transcripts[i]
                                for i, polygon in enumerate(text_polys)
                                if polygon.is_partly_within_image(image=im)
                            ]
                            im = transformed_im
                            break
                        max_tries -= 1

                    if max_tries == 0:
                        # loguru.logger.debug('Max tries has reached.')
                        return self.__getitem__(np.random.randint(0, len(self)))
                else:

                    polygon_list = []
                    for i in range(text_polys.shape[0]):
                        polygon_list.append(Polygon(text_polys[i].tolist()))

                    text_polys = PolygonsOnImage(polygon_list, shape=im.shape)

                polys = np.stack([poly.coords for poly in text_polys])

                (
                    score_map,
                    geo_map,
                    training_mask,
                    rectangles,
                    rois,
                ) = data_utils.get_score_geo(
                    im, polys, np.ones(polys.shape[0]), self.scale, self.size
                )

                # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
                image = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
                assert len(transcripts) == len(rectangles)

                if len(transcripts) == 0:
                    raise RuntimeError("No text found.")

                if self.vis:
                    self.visualize(
                        image=image,
                        polygons=rectangles,
                        score_map=score_map,
                        training_mask=training_mask,
                        image_name=image_path.stem,
                    )

                transcripts = str_label_converter.encode(transcripts)

                image = normalize_image(image)

                return (
                    image_path.as_posix(),
                    image,
                    score_map,
                    geo_map,
                    training_mask,
                    transcripts,
                    rectangles,
                    rois,
                )
            else:
                return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))
        except Exception as e:
            raise e
            # loguru.logger.warning('Something wrong with data processing. Resample.')
            # print("ICDAR DataLoader: Something wrong with data processing. Resample.")
            # return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))

    def __len__(self):
        return len(self.images)