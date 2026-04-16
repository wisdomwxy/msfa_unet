import os
import time

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU, show_results


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback:
    def __init__(
        self,
        net,
        input_shape,
        num_classes,
        image_ids,
        dataset_path,
        log_dir,
        cuda,
        miou_out_path=".temp_miou_out",
        eval_flag=True,
        period=1,
        name_classes=None,
        metrics_txt_only=False,
    ):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period
        self.name_classes = name_classes
        self.metrics_txt_only = metrics_txt_only

        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag and not self.metrics_txt_only:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), "a") as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(
            image, (self.input_shape[1], self.input_shape[0])
        )
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)
            ),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            if self.cuda:
                torch.cuda.synchronize()
            inference_start_time = time.time()

            pr = self.net(images)[0]

            if self.cuda:
                torch.cuda.synchronize()
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[
                int((self.input_shape[0] - nh) // 2) : int(
                    (self.input_shape[0] - nh) // 2 + nh
                ),
                int((self.input_shape[1] - nw) // 2) : int(
                    (self.input_shape[1] - nw) // 2 + nw
                ),
            ]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image, inference_time
    
    def save_detailed_metrics(self, hist, IoUs, PA_Recall, Precision, F1_Score, epoch):
        from .utils_metrics import per_Accuracy

        overall_accuracy = per_Accuracy(hist)
        txt_path = os.path.join(self.log_dir, "detailed_metrics.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Per-class metrics\n")
            f.write("=" * 60 + "\n")
            f.write(f"Epoch: {epoch}\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"{'Class':<15} {'mIoU':<10} {'Recall':<10} {'Precision':<10} {'F1-Score':<10}\n"
            )
            f.write("-" * 60 + "\n")
            for i in range(self.num_classes):
                class_name = self.name_classes[i] if self.name_classes else f"Class_{i}"
                f.write(
                    f"{class_name:<15} {IoUs[i]*100:<9.2f}% {PA_Recall[i]*100:<9.2f}% "
                    f"{Precision[i]*100:<9.2f}% {F1_Score[i]*100:<9.2f}%\n"
                )
            f.write("-" * 60 + "\n")
            f.write(
                f"{'Mean':<15} {np.nanmean(IoUs)*100:<9.2f}% {np.nanmean(PA_Recall)*100:<9.2f}% "
                f"{np.nanmean(Precision)*100:<9.2f}% {np.nanmean(F1_Score)*100:<9.2f}%\n"
            )
            f.write(f"Overall accuracy: {overall_accuracy*100:.2f}%\n")
            f.write("=" * 60 + "\n")

        if self.metrics_txt_only:
            with open(txt_path, "r", encoding="utf-8") as f:
                print(f.read(), end="")
            print(f"Saved: {txt_path}")
            return

        import csv

        with open(
            os.path.join(self.log_dir, "detailed_metrics.csv"), "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "mIoU(%)", "Recall(%)", "Precision(%)", "F1-Score(%)"])
            for i in range(self.num_classes):
                class_name = self.name_classes[i] if self.name_classes else f"Class_{i}"
                writer.writerow(
                    [
                        class_name,
                        f"{IoUs[i]*100:.2f}",
                        f"{PA_Recall[i]*100:.2f}",
                        f"{Precision[i]*100:.2f}",
                        f"{F1_Score[i]*100:.2f}",
                    ]
                )
            writer.writerow(
                [
                    "Mean",
                    f"{np.nanmean(IoUs)*100:.2f}",
                    f"{np.nanmean(PA_Recall)*100:.2f}",
                    f"{np.nanmean(Precision)*100:.2f}",
                    f"{np.nanmean(F1_Score)*100:.2f}",
                ]
            )
            writer.writerow(["Overall accuracy", f"{overall_accuracy*100:.2f}", "", "", ""])

        print("Saved metrics:")
        print(f"  - {txt_path}")
        print(f"  - {os.path.join(self.log_dir, 'detailed_metrics.csv')}")
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, "detection-results")
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            if not self.metrics_txt_only:
                print("Running validation (mIoU)...")
            else:
                print("Running validation...")

            if self.cuda:
                torch.cuda.synchronize()
            start_time = time.time()
            inference_times = []

            for image_id in tqdm(self.image_ids, disable=self.metrics_txt_only):
                image_path = os.path.join(
                    self.dataset_path, "VOC2007/JPEGImages", image_id + ".jpg"
                )
                image = Image.open(image_path)
                image, inference_time = self.get_miou_png(image)
                inference_times.append(inference_time)
                image.save(os.path.join(pred_dir, image_id + ".png"))

            if self.cuda:
                torch.cuda.synchronize()
            end_time = time.time()

            total_time = end_time - start_time
            total_images = len(self.image_ids)
            full_fps = total_images / total_time
            avg_time_per_image = total_time / total_images * 1000

            total_inference_time = sum(inference_times)
            pure_inference_fps = total_images / total_inference_time
            avg_inference_time = total_inference_time / total_images * 1000

            if not self.metrics_txt_only:
                print(f"  Inference FPS: {pure_inference_fps:.2f}")
                print("Computing mIoU...")
            miou_verbose = not self.metrics_txt_only
            hist, IoUs, PA_Recall, Precision, F1_Score = compute_mIoU(
                gt_dir,
                pred_dir,
                self.image_ids,
                self.num_classes,
                self.name_classes,
                verbose=miou_verbose,
            )
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            if not self.metrics_txt_only:
                with open(os.path.join(self.log_dir, "epoch_miou.txt"), "a") as f:
                    f.write(str(temp_miou))
                    f.write("\n")

            if self.metrics_txt_only:
                print(
                    f"Inference FPS: {pure_inference_fps:.2f}  "
                    f"(images: {total_images})\n"
                )
            self.save_detailed_metrics(
                hist, IoUs, PA_Recall, Precision, F1_Score, epoch
            )

            if self.metrics_txt_only:
                shutil.rmtree(self.miou_out_path)
                return

            if self.name_classes is not None:
                results_dir = os.path.join(self.log_dir, "detailed_results")
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                show_results(
                    results_dir,
                    hist,
                    IoUs,
                    PA_Recall,
                    Precision,
                    F1_Score,
                    self.name_classes,
                )

            with open(os.path.join(self.log_dir, "fps_result.txt"), "w") as f:
                f.write("FPS summary\n")
                f.write(f"Images: {total_images}\n")
                f.write(f"Total time (s): {total_time:.4f}\n")
                f.write(f"Time per image (ms): {avg_time_per_image:.2f}\n")
                f.write(f"End-to-end FPS: {full_fps:.2f}\n")
                f.write(f"Inference time (s): {total_inference_time:.4f}\n")
                f.write(f"Avg inference per image (ms): {avg_inference_time:.2f}\n")
                f.write(f"Inference FPS: {pure_inference_fps:.2f}\n")

            plt.figure()
            plt.plot(self.epoches, self.mious, "red", linewidth=2, label="train miou")
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("mIoU")
            plt.title("mIoU curve")
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Validation finished.")
            shutil.rmtree(self.miou_out_path)
