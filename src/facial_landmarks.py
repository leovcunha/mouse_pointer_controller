from model import Parent_Model


class Facial_Landmarks(Parent_Model):
    '''
    Class for the Facial Landmarks Model.
    '''

    def preprocess_output(self, outputs):
        '''
        Preprocess the output before feeding this model to the next model.
        '''
        outputs = outputs[self.output_name]
        #raise NotImplementedError
        coord = outputs[0]
        # print(coord.shape)
        left_x = int(coord[0] * self.image.shape[1])
        left_y = int(coord[1] * self.image.shape[0])
        right_x = int(coord[2] * self.image.shape[1])
        right_y = int(coord[3] * self.image.shape[0])
        left_eye = self.image[left_y-16:left_y+16, left_x-16:left_x+16]
        right_eye = self.image[right_y-16:right_y+16, right_x-16:right_x+16]

        return left_eye, left_x, left_y, right_eye, right_x, right_y
