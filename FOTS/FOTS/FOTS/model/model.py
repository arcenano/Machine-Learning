
import pathlib  # Paths
import typing  # Polygons list type
import numpy as np  # Images handling
import torch  # Activation functions
import torch.optim as optim  # Optimizer
from torch.optim.lr_scheduler import StepLR  # Scheduler
from pytorch_lightning import LightningModule  # Model module
import imgaug.augmentables.polys as ia_polys  # Polygons
import imgaug.augmentables.segmaps as ia_segmaps  # Score map visualization
import cv2  # Image creation for visualizing inputs and outputs
from .modules.shared_conv import SharedConv  # Shared convolutions
from .loss import DetectionLoss  # Detection loss function
from .modules.resnet import resnet50  # Resnet50 Backbone
from .modules.detector import Detector # Text Detector Branch
from ..utils.detect import * # Predict bounding boxes from score map
from .metrics import Metrics
from ..utils.util import keys
from .loss import FOTSLoss
from ..utils.post_processor import PostProcessor
from .modules.recognizer import Recognizer # Text Detector Branch
from ..rroi_align.functions.rroi_align import RRoiAlignFunction

class FOTSModel(LightningModule):
    def __init__(self, config):
        super(FOTSModel, self).__init__()
        self.config = config
        
        self.training = config.training

        self.mode = config.model.mode

        bbNet = resnet50(
            pretrained=True, weights=config.backbone_weights
        )  # Load resnet50 using the weights with path specified in the config file
        self.sharedConv = SharedConv(
            bbNet, config
        )  # Runs Restnet 50 and other convolutions with uplsampling and ReLUs
        self.detector = Detector(config)

        nclass = len(keys) + 2
        self.recognizer = Recognizer(nclass, config)
        self.roirotate = RRoiAlignFunction()

        self.pooled_height = 8
        self.spatial_scale = 1.0

        self.max_transcripts_pre_batch = (
            self.config.data_loader.max_transcripts_pre_batch
        )

        # self.loss = DetectionLoss()
        self.loss = FOTSLoss(config=config)

        self.metrics = Metrics()

        self.postprocessor = PostProcessor()

    def forward(self, images, boxes=None, rois=None):
        feature_map = self.sharedConv.forward(images)

        score_map, geo_map = self.detector(feature_map)

        if self.training:
            if self.mode == 'detection':

                data = dict(
                    score_maps=score_map,
                    geo_maps=geo_map,
                    transcripts=(None, None),
                    bboxes=boxes,
                    mapping=None,
                    indices=None,
                )
                return data

            # there are some hard samples, ###

            sampled_indices = torch.randperm(rois.size(0))[:self.max_transcripts_pre_batch]
            rois = rois[sampled_indices]

            ratios = rois[:, 4] / rois[:, 3]
            maxratio = ratios.max().item()
            pooled_width = np.ceil(self.pooled_height * maxratio).astype(int)

            roi_features = self.roirotate.apply(feature_map, rois, self.pooled_height, pooled_width, self.spatial_scale)
            lengths = torch.ceil(self.pooled_height * ratios)

            pred_mapping = rois[:, 0]
            pred_boxes = boxes

            preds = self.recognizer(roi_features, lengths.cpu())
            preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

            data = dict(score_maps=score_map,
                        geo_maps=geo_map,
                        transcripts=(preds, lengths),
                        bboxes=pred_boxes,
                        mapping=pred_mapping,
                        indices=sampled_indices)
            return data

        else:

            score = score_map.cpu().numpy()
            geometry = geo_map.cpu().numpy()

            pred_boxes = []
            rois = []
            for i in range(score.shape[0]):
                s = score[i]
                g = geometry[i]
                bb = get_boxes(s, g, score_thresh=0.9)
                # print(bb.shape())
                if bb is not None:
                    roi = []
                    for _, gt in enumerate(bb[:, :8].reshape(-1, 4, 2)):

                        #print("GT: \n", gt)

                        rr = cv2.minAreaRect(gt)
                        center = rr[0]
                        (w, h) = rr[1]
                        min_rect_angle = rr[2]

                        if h > w:
                            min_rect_angle = min_rect_angle + 180
                            roi.append([i, center[0], center[1], w, h, -min_rect_angle])
                        else:
                            roi.append([i, center[0], center[1], h, w, -min_rect_angle])

                    pred_boxes.append(bb)
                    rois.append(np.stack(roi))

            if self.mode == "detection":

                if len(pred_boxes) > 0:
                    pred_boxes = torch.as_tensor(
                        np.concatenate(pred_boxes),
                        dtype=feature_map.dtype,
                        device=feature_map.device,
                    )
                    rois = torch.as_tensor(
                        np.concatenate(rois),
                        dtype=feature_map.dtype,
                        device=feature_map.device,
                    )
                    pred_mapping = rois[:, 0]
                else:
                    pred_boxes = None
                    pred_mapping = None

                data = dict(
                    score_maps=score_map,
                    geo_maps=geo_map,
                    transcripts=(None, None),
                    bboxes=pred_boxes,
                    mapping=pred_mapping,
                    indices=None,
                )
                return data

            if len(rois) > 0:
                pred_boxes = torch.as_tensor(
                    np.concatenate(pred_boxes),
                    dtype=feature_map.dtype,
                    device=feature_map.device,
                )
                rois = torch.as_tensor(
                    np.concatenate(rois),
                    dtype=feature_map.dtype,
                    device=feature_map.device,
                )
                pred_mapping = rois[:, 0]

                ratios = rois[:, 4] / rois[:, 3]
                maxratio = ratios.max().item()
                pooled_width = np.ceil(self.pooled_height * maxratio).astype(int)
                roi_features = self.roirotate.apply(
                    feature_map,
                    rois,
                    self.pooled_height,
                    pooled_width,
                    self.spatial_scale,
                )

                lengths = torch.ceil(self.pooled_height * ratios)

                preds = self.recognizer(roi_features, lengths.cpu())
                preds = preds.permute(1, 0, 2)  # B, T, C -> T, B, C

                data = dict(
                    score_maps=score_map,
                    geo_maps=geo_map,
                    transcripts=(preds, lengths),
                    bboxes=pred_boxes,
                    mapping=pred_mapping,
                    indices=None,
                )
                return data
            else:
                data = dict(
                    score_maps=score_map,
                    geo_maps=geo_map,
                    transcripts=(None, None),
                    bboxes=None,
                    mapping=None,
                    indices=None,
                )

                return data

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer_type)( # Get the desired optimizer from the torch optimizers
            self.parameters(), **self.config.optimizer # Pass in model parameters and learning rate
        )

        if not self.config.lr_scheduler.name:
            return optimizer
        else:
            if self.config.lr_scheduler.name == "StepLR":
                lr_scheduler = StepLR(optimizer, **self.config.lr_scheduler.args)
            else:
                raise NotImplementedError()
            return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)     

    """Create lightning module functions """

    # def training_step(self, *args, **kwargs):
    #     input_data = args[0]
    #     bboxes = input_data["bboxes"]
    #     rois = input_data["rois"]

    #     output = self.forward(images=input_data["images"], boxes=bboxes, rois=rois)

    #     # print(input_data['score_maps'])

    #     # print("\n\nInput Data: ", input_data)
    #     # print("\n\n")

    #     # if input_data["training_masks"]

    #     self.visualize(
    #         input_data["image_names"],
    #         input_data["images"],
    #         bboxes,
    #         output["score_maps"],
    #         output["geo_maps"],
    #         mapping=input_data["mapping"],
    #     )

    #     reg_loss, cls_loss, iou = self.loss(
    #         gt_score=input_data["score_maps"],
    #         pred_score=output["score_maps"],
    #         gt_geo=input_data["geo_maps"],
    #         pred_geo=output["geo_maps"],
    #         ignored_map=input_data["training_masks"],
    #     )

    #     accuracy, confusion, dice , pixel_acc = self.metrics.get_metrics(
    #     output["score_maps"], input_data["score_maps"].to(torch.int64))

    #     loss = reg_loss + cls_loss

    #     # preds = [
    #     #     dict(
    #     #         boxes =  output["score_maps"],
    #     #         scores = 
    #     #         labels = 

    #     #     )
    #     # ] 

    #     # map = MAP()

    #     # map.update()

    #     self.log("iou", iou, logger=True)
    #     self.log("accuracy", accuracy, logger=True, prog_bar=True)
    #     #self.log("confusion", confusion, logger=True, prog_bar=True)
    #     self.log("dice", dice, logger=True, prog_bar=True)
    #     self.log("pixel", pixel_acc, logger=True, prog_bar=True)
    #     self.log("loss", loss, logger=True)
    #     self.log("reg_loss", reg_loss, logger=True, prog_bar=True)
    #     self.log("cls_loss", cls_loss, logger=True, prog_bar=True)

    #     return loss


    def training_step(self, *args, **kwargs):
        input_data = args[0]
        bboxes = input_data['bboxes']
        rois = input_data['rois']

        output = self.forward(images=input_data['images'],
                              boxes=bboxes,
                              rois=rois)

        sampled_indices = output['indices']
        y_true_recog = (input_data['transcripts'][0][sampled_indices],
                        input_data['transcripts'][1][sampled_indices])

        loss_dict = self.loss(y_true_cls=input_data['score_maps'],
                              y_pred_cls=output['score_maps'],
                              y_true_geo=input_data['geo_maps'],
                              y_pred_geo=output['geo_maps'],
                              y_true_recog=y_true_recog,
                              y_pred_recog=output['transcripts'],
                              training_mask=input_data['training_masks'])

        loss = loss_dict['reg_loss'] + loss_dict['cls_loss'] + loss_dict['recog_loss']
        self.log('loss', loss, logger=True)
        self.log('reg_loss', loss_dict['reg_loss'], logger=True, prog_bar=True)
        self.log('cls_loss', loss_dict['cls_loss'], logger=True, prog_bar=True)
        self.log('recog_loss', loss_dict['recog_loss'], logger=True, prog_bar=True)

        return loss

    def validation_step(self, *args, **kwargs):
        input_data = args[0]
        output = self.forward(images=input_data['images'])
        output['images_names'] = input_data['image_names']
        return output

    def validation_step_end(self, *args, **kwargs):
        output: dict = args[0]

        boxes_list = []
        transcripts_list = []

        pred_boxes = output['bboxes']
        pred_mapping = output['mapping']
        image_names = output['images_names']

        self.log('pred_boxes', pred_boxes.shape[0] if pred_boxes is not None else 0, prog_bar=True)

        if pred_boxes is None:
            return dict(image_names=image_names, boxes_list=boxes_list, transcripts_list=transcripts_list)

        # pred_transcripts, pred_lengths = output['transcripts']
        # indices = output['indices']
        # # restore order
        # # pred_transcripts = pred_transcripts[:, ::-1, :][indices[::-1]]
        # # pred_lengths = pred_lengths[:, ::-1, :][indices[::-1]]
        #
        # for index in range(len(image_names)):
        #     selected_indices = np.argwhere(pred_mapping == index)
        #     boxes = pred_boxes[selected_indices]
        #     transcripts = pred_transcripts[:, selected_indices, ]
        #     lengths = pred_lengths[selected_indices]
        #     boxes, transcripts = self.postprocessor(boxes=boxes, transcripts=(transcripts, lengths))
        #     boxes_list.append(boxes)
        #     transcripts_list.append(transcripts)
        #     image_path = '/media/mydisk/ocr/det/icdar2015/detection/test/imgs/' + image_names[index]
        #     visualize(image_path, boxes, transcripts)
        #
        #     # visualize_box
        #
        # return dict(image_names=image_names, boxes_list=boxes_list, transcripts_list=transcripts_list)

    # def validation_step(self, *args, **kwargs):
    #     input_data = args[0]
    #     bboxes = input_data["bboxes"]
    #     rois = input_data["rois"]

    #     output = self.forward(images=input_data["images"], boxes=bboxes, rois=rois)

    #     reg_loss, cls_loss, iou= self.loss(
    #         gt_score=input_data["score_maps"],
    #         pred_score=output["score_maps"],
    #         gt_geo=input_data["geo_maps"],
    #         pred_geo=output["geo_maps"],
    #         ignored_map=input_data["training_masks"],
    #     )

    #     accuracy, confusion, dice, pixel_acc = self.metrics.get_metrics(
    #     output["score_maps"], input_data["score_maps"].to(torch.int64))
        
    #     loss = reg_loss + cls_loss  
        
    #     self.log("iou", iou, logger=True)
    #     self.log("accuracy", accuracy, logger=True, prog_bar=True)
    #     #self.log("confusion", confusion, logger=True, prog_bar=True)
    #     self.log("dice", dice, logger=True, prog_bar=True)
    #     self.log("pixel", pixel_acc, logger=True, prog_bar=True)
    #     self.log("loss", loss, logger=True)
    #     self.log("reg_loss", reg_loss, logger=True, prog_bar=True)
    #     self.log("cls_loss", cls_loss, logger=True, prog_bar=True)
 
    def visualize(
        self,
        image_names: str,
        image: np.ndarray,
        polygons: typing.List[typing.List[float]],
        score_map: np.ndarray,
        geo_map,
        mapping,
    ):

        # print(
        #     f"Image Names: {image_names}\nInput Image Dimensions: {image.shape}\nPolygons Dimensions: {polygons.shape}\nScore Maps dimensions: {score_map.shape}\nGeo Maps Dimensions: {geo_map.shape}"
        # )

        # Get tensors on cpu, remove gradient propagation and convert to numpy
        image = image.cpu().detach().numpy()
        polygons = polygons.cpu().detach().numpy()
        score_maps = score_map.cpu().detach().numpy()
        geo_maps = geo_map.cpu().detach().numpy()
        mapping = mapping.cpu().detach().numpy()

        for i in range(image.shape[0]):  # iterate over batch size

            # Select image name and create path
            image_name = image_names[i]
            path = pathlib.Path(image_name)

            cv2.imwrite(
                self.config.images + path.name + f"_test.jpg",
                image[i].transpose(1, 2, 0).astype(dtype=np.uint8),
            )

            score_map = score_maps[i][0] > 0.8

            # Get Heat Map
            score_map = ia_segmaps.SegmentationMapsOnImage(
                score_map.astype(dtype=np.uint8),
                shape=score_map.shape,
            )

            score_map = score_map.resize(
                sizes=(image[i].shape[2], image[i].shape[1]), interpolation="nearest"
            )

            new_image = score_map.draw_on_image(
                image[i].transpose(1, 2, 0).astype(dtype=np.uint8)
            )

            cv2.imwrite(self.config.images + path.name + f"_score.jpg", new_image[0])

            # Select current image
            img = image[i]

            # Polygons array
            polys = []

            # Loop through all polygons on image
            for n in range(len(mapping)):
                if mapping[n] == i:
                    # Fill polygon array for image
                    polys.append(
                        ia_polys.Polygon(polygons[n])
                    )  # .tolist()))#np.array(polygons[i]).reshape(4,2)))

            # polygon_list = []
            # for polygon in polygons[i, :]:
            #     polygon_list.append(ia_polys.Polygon(np.array(polygon).reshape(4, 2))):

            polygons_on_image = ia_polys.PolygonsOnImage(
                polygons=polys, shape=img.shape  # image[i, :, :].shape , polygons[i, :]
            )

            # print('=============\n\n')
            # print(polygons_on_image.shape)
            # print(img.shape)
            # print('\n\n=============')

            tempimg = img

            new_image = polygons_on_image.draw_on_image(
                tempimg.transpose(1, 2, 0).astype(dtype=np.uint8)
            )  # image[i, :, :]

            cv2.imwrite(
                self.config.images + path.name + "_polygons_" + f"{i}.jpg", new_image
            )  # path.name + image_name

            # for j in range(len(score_maps.shape[1])):

            #     score_map = ia_segmaps.SegmentationMapsOnImage(
            #         score_maps[i, j, :, :].astype(dtype=np.uint8),
            #         shape=image[i, :, :].shape,
            #     )
            #     new_image = score_map.draw_on_image(
            #         image[i, :, :].astype(dtype=np.uint8)
            #     )
            #     cv2.imwrite(image_name + f"_score{i}_{j}.jpg", new_image[0])


