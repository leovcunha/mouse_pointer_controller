'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import Parent_Model
import cv2


class Facial_Landmarks(Parent_Model):
    '''
    Class for the Facial Landmarks Model.
    '''

    def preprocess_output(self, outputs):
        '''
        Preprocess the output before feeding this model to the next model.
        '''
        #raise NotImplementedError
        coord = outputs[0]
        # print(coord.shape)
        left_x = int(coord[0] * self.image.shape[1])
        left_y = int(coord[1] * self.image.shape[0])
        right_x = int(coord[2] * self.image.shape[1])
        right_y = int(coord[3] * self.image.shape[0])

        return left_x, left_y, right_x, right_y
