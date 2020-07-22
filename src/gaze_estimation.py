from model import Parent_Model
import cv2


class Gaze_Estimation(Parent_Model):
    '''
    Class for the Gaze Estimation.
    '''

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):

        left_preprocessed = self.preprocess_input(
            left_eye_image.copy(), 60, 60)
        right_preprocessed = self.preprocess_input(
            right_eye_image.copy(), 60, 60)
        input_dict = {'head_pose_angles': head_pose_angle,
                      'left_eye_image': left_preprocessed, 'right_eye_image': right_preprocessed}
        outputs = self.net.infer(input_dict)

        return self.preprocess_output(outputs)

    def preprocess_output(self, outputs):
        '''
        Preprocess the output before feeding this model to the next model.
        '''
        #raise NotImplementedError
        outputs = outputs[self.output_name]

        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]

        return x, y, z
