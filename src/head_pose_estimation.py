from model import Parent_Model


class Head_Pose_Estimation(Parent_Model):
    '''
    Class for the Head Pose Estimation.
    '''

    def preprocess_output(self, outputs):
        '''
        Preprocess the output before feeding this model to the next model.
        '''
        #raise NotImplementedError
        yaw = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]

        return yaw, pitch, roll
