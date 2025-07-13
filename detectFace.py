import warnings
warnings.filterwarnings('ignore')
import cv2
import mediapipe as mp
import math
import numpy as np
import os
from functools import reduce
from deepface import DeepFace
import matplotlib.pyplot as plt



class FaceDetect:

    def __init__(self):
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.bboxColor=(0,0,256)
        self.bboxThickness = 4
        self.labelFont = cv2.FONT_HERSHEY_SIMPLEX
        self.labelFontScale = 2
        self.labelColor = (255, 0, 0)
        self.labelThickness = 3
        self.labelPosArg = 20

    @staticmethod
    def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))
        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            return None
        
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def detect(self, image):
        # image = cv2.imread(image_file)
        image_rows, image_cols, _ = image.shape
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            # bboxes = []
            coords = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                bbox = dict(xmin=bbox.xmin, ymin=bbox.ymin, width=bbox.width, height=bbox.height)
                # bboxes.append(bbox)
                rect_start_point = self._normalized_to_pixel_coordinates(bbox['xmin'], bbox['ymin'], image_cols, image_rows)
                rect_end_point = self._normalized_to_pixel_coordinates(bbox['xmin'] + bbox['width'],
                                                                  bbox['ymin'] + bbox['height'], 
                                                                  image_cols,image_rows)
                coords.append((rect_start_point, rect_end_point))
            coords = dict(enumerate(coords))
        else:
            coords = None
            
        return image, coords
        # return image, dict(enumerate(list(zip(bboxes, coords))))

    
    def annotate(self, image):
        # image = cv2.imread(image_file)
        annotated_image = image.copy()
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                self.dete = detection
                self.mp_drawing.draw_detection(annotated_image, detection)
        return annotated_image

    
    def drawBbox(self, image):
        image, bbox_coords = self.detect(image)
        annotated_image = image.copy()
        if bbox_coords:
            for c in bbox_coords.values():
                annotated_image = cv2.rectangle(annotated_image, c[1], c[0], color=self.bboxColor,
                                             thickness=self.bboxThickness)
        return annotated_image


    def crop(self, image):
        image, bbox_coords = self.detect(image)
        cropped_images = {}
        try:
            for key, value in bbox_coords.items():
                start = value[0]
                end = value[1]
                cropped_images[key] = image[start[1]:end[1],start[0]:end[0]]
            return cropped_images
        except:
            pass
        
        
    def label(self, image):
        image, coords = self.detect(image)
        annotated_image = image.copy()
        try:
            if coords:
                for key, value in coords.items():
                    pos = tuple(np.array(value[0])-self.labelPosArg)
                    annotated_image = cv2.putText(annotated_image, str(f'Person {int(key)+1}'), pos, self.labelFont, 
                           self.labelFontScale, self.labelColor, self.labelThickness, cv2.LINE_AA)
        except:
            pass
        return annotated_image






class EmotionAnalyze(FaceDetect):

    def __init__(self):
        super().__init__()

    def detectEmotionOfFace(self, image):
        emotions = DeepFace.analyze(image, actions = ['emotion'], detector_backend='mediapipe')[0]['emotion']
        emotions = dict(sorted(list(emotions.items()), key=lambda x:x[1], reverse=True))
        return emotions

    def labelEmotion(self, image):
        image, coords = self.detect(image)
        annotated_image = image.copy()
        cropped_images = self.crop(annotated_image)
        emots = {}
        try:
            if coords:
                for key, value in coords.items():
                    pos = tuple(np.array(value[0])-self.labelPosArg)
                    
                    emotions = self.detectEmotionOfFace(cropped_images[key])
                    emotion_label =  f" : {list(emotions.keys())[0].capitalize()}"
                    annotated_image = cv2.putText(annotated_image,
                                                  str(f'Person {int(key)+1}'+emotion_label), pos, self.labelFont, 
                           self.labelFontScale, self.labelColor, self.labelThickness, cv2.LINE_AA)
                    emots[key] = emotions
        except:
            emots = None
        return annotated_image, emots





def isolateImages(croppedImages, imageLocation):
    for idx in list(set(reduce(lambda a,b: a+b, [list(i.keys()) for i in croppedImages]))):
        if f'person{idx+1}' not in os.listdir(imageLocation):
            os.mkdir(os.path.join(imageLocation, f'person{idx+1}'))    
    for idx,imgDic in enumerate(croppedImages):
        if imgDic:
            for key,value in imgDic.items():
                cv2.imwrite(f'./crops/person{key+1}/{idx}.png', value)





class emotionVisualize:
    
    def __init__(self,dim=500):
        self.PLANE_DIM = dim
        self.plane = self.create_plane()
        self.line_angles = {'angry': 0,
                         'happy': 45,
                         'sad': 90,
                         'neutral': 135,
                         'fear': 180,
                         'surprise': 225,
                         'disgust': 270,
                         'inconclusive': 315}
        self.figsize_percent = 0.77
        self.figsize_text_ratio = 1.5
        self.line_thickness_bias = 0
        
    def create_plane(self):
        self.plane = np.zeros((self.PLANE_DIM, self.PLANE_DIM, 3), np.uint8) + 255
        self.origin_pt = (int(self.PLANE_DIM/2), int(self.PLANE_DIM/2))
        #self.plane = cv2.circle(self.plane.copy(), self.origin_pt, int(self.PLANE_DIM/2), (255,0,0), 1)
    
    def pt2(self,angle,length_percent=None):
        if length_percent == None:
            hyp = self.PLANE_DIM/2
        else:
            hyp = length_percent * self.PLANE_DIM/2
        hyp = self.figsize_percent * hyp
        base = hyp * np.cos(np.radians(angle))
        perp = hyp * np.sin(np.radians(angle))
        return (int(self.origin_pt[0]+base), int(self.origin_pt[1]+perp))
    
    def draw_lines(self):
        for value in self.line_angles.values():
             self.plane = cv2.line(self.plane.copy(), self.origin_pt, self.pt2(value), (0,255,0),
                                   math.ceil(self.figsize_percent)+round(self.line_thickness_bias))
    
    def draw_emotion_graph(self,probs:list):
        # assert isinstance(probs, np.ndarray)
        # assert round(probs.sum()) == 1   
        pts = [self.pt2(i,j) for i,j in zip(self.line_angles.values(),probs.tolist())]
        for index in range(len(pts)-1):
            self.plane = cv2.line(self.plane.copy(), pts[index], pts[index+1], (0,0,255),
                                  math.ceil(self.figsize_percent)+round(self.line_thickness_bias))
        self.plane = cv2.line(self.plane.copy(), pts[0], pts[-1], (0,0,255),
                              math.ceil(self.figsize_percent)+round(self.line_thickness_bias))
        maximum_proba = []
        for key,value,prob in zip(self.line_angles.keys(),[self.pt2(ang) for ang in self.line_angles.values()],probs):
            fontScale = 2*(self.PLANE_DIM*self.figsize_percent/self.figsize_text_ratio)/500
            if prob == probs.max():
                maximum_proba.append((key,prob))
                thickness = round(self.figsize_percent)+round(self.line_thickness_bias)+1
                color = (124, 73, 196)
                fontScale *= 1.2
            else:
                thickness = round(self.figsize_percent)+round(self.line_thickness_bias)
                color = (0,0,0)
            self.plane = cv2.putText(self.plane.copy(), key, value, cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale, color, thickness)
        return maximum_proba
            
    
    def visualize(self, person, probs):
        # print(probs)
        emotions = list(self.line_angles.keys())[:-1]
        probs = [probs[i]/100 for i in emotions]
        # print(probs)


        def add_inconclusivity(probs):
            prob_greater_than_5 = any([i>=0.5 for i in probs])
    
            if prob_greater_than_5:
                probs = np.array(list(probs) + [0.0])
            else:
                probs = np.array(list(probs) + [1.0])
            return probs

        probs = add_inconclusivity(probs)
        
        self.create_plane()
        self.draw_lines()
        maximum_proba = self.draw_emotion_graph(probs)
        x = int(self.PLANE_DIM * (1-self.figsize_percent) * 0.05)
        y = int(self.PLANE_DIM * (1-self.figsize_percent) * 0.2)
        fontScale = 3*(self.PLANE_DIM*self.figsize_percent/self.figsize_text_ratio)/500
        thickness = round(self.figsize_percent)+round(self.line_thickness_bias)+1
        color = (168, 74, 50)
        index = y
        for emotion,prob in maximum_proba:
            if isinstance(person, int):
                person += 1
            self.plane = cv2.putText(self.plane.copy(), f"Person {person} : {emotion} : {np.round(prob*100, 2)}%",
                                     (x,index), cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale, color, thickness)
            index += y