from model import Parent_Model


class Gaze_Estimation(Parent_Model):
    '''
    Class for the Gaze Estimation.
    '''

    def preprocess_output(self, outputs):
        '''
        Preprocess the output before feeding this model to the next model.
        '''
        #raise NotImplementedError
        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]

        return x, y, z
