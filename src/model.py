import cv2
import logging as log
from openvino.inference_engine import IECore


class Parent_Model:
    '''
    Class for the Parent Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Sets instance variables.
        '''
        #raise NotImplementedError
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.model_name = model_name

        try:
            self.core = IECore()
            if extensions and "CPU" in self.device:
                self.core.add_extension(extensions, device)
            self.model = self.core.read_network(
                model=self.model_structure, weights=self.model_weights)

        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        loads the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        #raise NotImplementedError
        self.check_model()

        self.net = self.core.load_network(
            network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        # NotImplementedError
        self.image = image.copy()
        input_img = self.preprocess_input(
            self.image, self.input_shape[3], self.input_shape[2])
        input_dict = {self.input_name: input_img}
        out = self.net.infer(input_dict)
        return self.preprocess_output(out)

    def check_model(self):
        #raise NotImplementedError
        supported_layers = self.core.query_network(self.model, self.device)
        not_supported_layers = [
            l for l in self.model.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("The following layers are not supported "
                      "by the IECore for the specified device {}:\n {}"
                      .format(self.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions ")
            raise NotImplementedError(
                "Some layers are not supported on the device")
        else:
            return True

    def preprocess_input(self, image, width, height):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #raise NotImplementedError
        preprocessed_image = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_AREA)
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape(
            1, *preprocessed_image.shape)
        return preprocessed_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
