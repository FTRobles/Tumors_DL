# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:05:46 2019

@author: daewo
"""
import colorsys
import random
import os

import numpy as np
import cv2

from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.segmentation import clear_border

from sklearn.metrics import confusion_matrix, roc_curve, auc

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon

class PostProcessing:
    
    #Post processing class image
    def postProcessImage(self,class_image):
        
        #Structure for opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        
        #Opening image t clean it and soften the shapes
        open_image = cv2.morphologyEx(class_image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                
        #Clear object touching the border of the image
        border_image = clear_border(open_image)
        
        #Clear objects that are not intersecting central window of the image
        [ren, col] = class_image.shape

        mask = np.zeros(class_image.shape, dtype=np.uint8)
        start = (round(ren/8), round(col/8))
        end = (round(ren - ren/8), round(col - col/8))
        rr, cc = rectangle(start, end=end, shape=mask.shape)
        mask[rr, cc] = 1
        
        #mask_idx = np.where(mask == 1)
        
        n_objs, labels = cv2.connectedComponents(border_image)
        
        clear_image = np.zeros(class_image.shape, dtype=np.uint8)
        
        for label in range(1,n_objs):
            #Get pixels that belong to the object
            intersection_idx = np.where(np.logical_and((labels == label),(mask == 1)))
            
            if intersection_idx[0].any():
                obj_idx = np.where(labels == label)
                clear_image[obj_idx[0],obj_idx[1]] = 1

#

        #Display original class image and opening
#        fig = plt.figure()
#        fig.subplots_adjust(hspace=0.4, wspace=0.4)
#        
#        ax = fig.add_subplot(1, 5, 1)
#        ax.imshow(class_image, cmap="gray")
#        
#        ax = fig.add_subplot(1, 5, 2)
#        ax.imshow(open_image, cmap="gray")
#        
#        ax = fig.add_subplot(1, 5, 3)
#        ax.imshow(border_image, cmap="gray")
#        
#        ax = fig.add_subplot(1, 5, 4)
#        ax.imshow(mask, cmap="gray")
#        
#        ax = fig.add_subplot(1, 5, 5)
#        ax.imshow(clear_image, cmap="gray")
        
        return clear_image
    
    # Get the object regions found in pixel classification    sorted by probability
    def getRegions(self,test_image,class_image,prob_image):
        #Obtain list of all objects found in the binary image
        n_objs, labels = cv2.connectedComponents(class_image)
        
        print("Number of possible tumor regions")
        print(n_objs)
        print(len(labels))
        
        #plt.imshow(labels)
        
        #Objects binary image, probability and bounding box
        obj_imgs = []
        probs = []
        boxes = []
        
        for label in range(0,n_objs):
            
            #Get pixels that belong to the object
            obj_idx = np.where(labels == label)
        
            #Construct binary image wiht only the object mask
            obj_img = np.zeros(test_image.shape)
            obj_img[obj_idx] = 1
            obj_imgs.append(obj_img)
            
            #Get probability of the object using the computed probability image
            obj_prob = prob_image[obj_idx]
            probs.append(np.mean(obj_prob))
            
            #Get bounding box using object mask
            x1 = np.min(obj_idx[1])
            y1 = np.min(obj_idx[0])
            x2 = np.max(obj_idx[1])
            y2 = np.max(obj_idx[0])
            boxes.append([x1, y1, x2, y2])
        
        #plt.show()
        
        n_objs -= 1
        obj_imgs = obj_imgs[1:]
        probs = probs[1:]
        boxes = boxes[1:]

        sorted_idx = sorted(range(len(probs)), key=lambda k: probs[k])
        sorted_idx.reverse()
        probs = np.array(probs)[sorted_idx]
        obj_imgs = np.array(obj_imgs)[sorted_idx]
        boxes = np.array(boxes)[sorted_idx]
        
        print(probs)
        
        return n_objs, obj_imgs, probs, boxes
    
    #function to generate random colors for object display  
    def random_colors(self,N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    # Function to apply a mask to each object in the image 
    def apply_mask(self,image, mask, color, alpha=0.2):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1.0,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c]*255,
                                      image[:, :, c])
        return image
    
    # Methods object visualization in image
    def createFig(self,n_objs, obj_imgs, probs, boxes, test_image):
        
        #Visualize objects
        fig, ax = plt.subplots(1,figsize = (9,8))
        
        ax.axis('off')
        
        # Generate random colors
        colors = self.random_colors(n_objs)
        
        #Transform test_image to RGB color (to display objects in different colors)
        masked_img = cv2.cvtColor(test_image.astype(np.float32),cv2.COLOR_GRAY2RGB)
        for i in range(0, n_objs):
        
            #Get new color for object
            color = colors[i-1]
        
            #Show bounding box of the object in the image
            box = boxes[i]
            p = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],linewidth=2,
                                            alpha=0.7, linestyle="dashed",
                                            edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
            #Show propability of the object in the image
            class_id = 1
            score = probs[i]
            label = "Tumor"
            caption = "{} {:.3f}".format(label, score) 
            ax.text(box[0], box[1], caption,
                    color='w', size=11, backgroundcolor="none")
        
            mask_img = obj_imgs[i] 
            masked_img = self.apply_mask(masked_img, mask_img, color)
        
            #Show mask of the object in the image
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask_img.shape[0] + 2, mask_img.shape[1] + 2), dtype=np.float32)
            padded_mask[1:-1, 1:-1] = mask_img
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
        
        #show results
        ax.imshow(masked_img.astype(np.float32))
        plt.show()
        
        return fig
    
    def visualize(self,test_images,class_images,prob_images,save_path="",fold=0,n_disp=0):
        
        for i in range(len(test_images)):
            
            test_image = test_images[i]
            class_image = class_images[i]
            prob_image = prob_images[i]
            
            #Save class and probability image
            if save_path != "":
                class_filename = "class_fold_" + str(fold) + "_" + str(i) + ".jpg"
                class_filename = os.path.join(save_path,"Class",class_filename)
                plt.imsave(class_filename, class_image, cmap="gray")
                
                prob_filename = "prob_fold_" + str(fold) + "_" + str(i) + ".jpg"
                prob_filename = os.path.join(save_path,"Prob",prob_filename)
                plt.imsave(prob_filename, prob_image, cmap="gray")
        
            #postprocessing open image to filter small objects
            pprocess_image = self.postProcessImage(class_image)
            
            #Save class and probability image
            if save_path != "":
                pprocess_filename = "pprocess_fold_" + str(fold) + "_" + str(i) + ".jpg"
                pprocess_filename = os.path.join(save_path,"Open",pprocess_filename)
                plt.imsave(pprocess_filename, pprocess_image, cmap="gray")
            
            #get image objects 
            [n_objs, obj_imgs, probs, boxes] = self.getRegions(test_image,pprocess_image,prob_image)
            
            if n_disp!=0:
                if n_disp<n_objs:
                    n_objs = n_disp
        
            #visualize objects
            fig = self.createFig(n_objs, obj_imgs, probs, boxes, test_image)
            
            if save_path != "":
                masked_filename = "masked_fold_" + str(fold) + "_" + str(i) + ".jpg"
                fig_filename = os.path.join(save_path,"Masked",masked_filename)
                plt.savefig(fig_filename)
        
        return fig
    
    def evaluateSegmentation(self,test_masks,class_images,save_path="",fold=0):
        
        tests = np.array([])
        classes = np.array([])
        accuracy = []
        sensitivity = []
        specificity = []
        roc_au = []
        
        for i in range(len(test_masks)):
            
            test_mask = test_masks[i].astype(bool)
            class_image = class_images[i].astype(bool)
            
            test_mask = test_mask.flatten()
            class_image = class_image.flatten()
            
            tests = np.concatenate((tests,test_mask),axis=0)
            classes =  np.concatenate((classes,class_image),axis=0)
            
            #Confusion matrix, Accuracy, sensitivity and specificity
            cm1 = confusion_matrix(test_mask,class_image)
            print('Confusion Matrix : \n', cm1)
            
            total1=sum(sum(cm1))
            
            #####from confusion matrix calculate accuracy
            acc = (cm1[0,0]+cm1[1,1])/total1
            accuracy.append(acc)
            print ('Accuracy : ', acc)
            
            sens = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            sensitivity.append(sens)
            print('Sensitivity : ', sens)
            
            spec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
            specificity.append(spec)
            print('Specificity : ', spec)
            
            fpr, tpr, _ = roc_curve(test_mask,class_image)
            au = auc(fpr, tpr)
            roc_au.append(au)
            print('AUC: ', au)
            
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % au)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
            
            if save_path != "":
                roc_filename = "roc_fold_" + str(fold) + "_" + str(i) + ".jpg"
                roc_filename = os.path.join(save_path,"ROC",roc_filename)
                plt.savefig(roc_filename)   
                
                results_filename = "results_fold_" + str(fold) + "_" + str(i) + ".txt"
                results_filename = os.path.join(save_path,"ROC",results_filename)
                results_file= open(results_filename,"w")
                results_file.write("Accuracy: " + str(acc) + "\n")
                results_file.write("Sensitivity: " + str(sens) + "\n")
                results_file.write("Specificity: " + str(spec) + "\n")
                results_file.write("AU-ROC: " + str(au))
                
                results_file.close()
        

        #Confusion matrix, Accuracy, sensitivity and specificity of all predictions
        cm_total = confusion_matrix(tests,classes)
        print('Confusion Matrix : \n', cm_total)
        
        total_total=sum(sum(cm_total))
            
        acc_total = (cm_total[0,0]+cm_total[1,1])/total_total
        accuracy.append(acc_total)
        print ('Accuracy total: ', acc_total)
        
        sens_total = cm_total[0,0]/(cm_total[0,0]+cm_total[0,1])
        sensitivity.append(sens_total)
        print('Sensitivity total: ', sens_total)
        
        spec_total = cm_total[1,1]/(cm_total[1,0]+cm_total[1,1])
        specificity.append(spec_total)
        print('Specificity total: ', spec_total)
        
        fpr_total, tpr_total, _ = roc_curve(tests,classes)
        au_total = auc(fpr_total, tpr_total)
        roc_au.append(au_total)
        print('AUC total: ', au_total)
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve Total (area = %0.2f)' % au_total)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
        if save_path != "":
            roc_filename = "roc_fold_" + str(fold) + "_total.jpg"
            roc_filename = os.path.join(save_path,"ROC",roc_filename)
            plt.savefig(roc_filename)   
            
            results_filename = "results_fold_" + str(fold) + "_total.txt"
            results_filename = os.path.join(save_path,"ROC",results_filename)
            results_file= open(results_filename,"w")
            results_file.write("Accuracy: " + str(acc_total) + "\n")
            results_file.write("Sensitivity: " + str(sens_total) + "\n")
            results_file.write("Specificity: " + str(spec_total) + "\n")
            results_file.write("AU-ROC: " + str(au_total))
            
            results_file.close()
            
        return [accuracy,sensitivity,specificity,roc_au]

