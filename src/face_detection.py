from model import Parent_Model
import cv2


class Face_Detection(Parent_Model):
    '''
    Class for the Face Detection Model.
    '''

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #raise NotImplementedError
        coords = []
        for box in outputs[0][0]:
            conf = box[2]
            if conf >= self.threshold:
                coords.append([box[3], box[4], box[5], box[6]])
        coord = coords[0]
        coord[0] = int(coord[0] * self.image.shape[1])
        coord[1] = int(coord[1] * self.image.shape[0])
        coord[2] = int(coord[2] * self.image.shape[1])
        coord[3] = int(coord[3] * self.image.shape[0])
        cv2.rectangle(self.image, (coord[0], coord[1]),
                      (coord[2], coord[3]), (0, 0, 255), 1)
        return self.image[coord[1]:coord[3], coord[0]:coord[2]]
