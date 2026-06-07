import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import numpy as np

class TRTWrapper:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")
        
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Allocate memory buffers for inputs and outputs
        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({'host': cuda.pagelocked_empty(trt.volume(shape), dtype), 
                                    'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': cuda.pagelocked_empty(trt.volume(shape), dtype), 
                                     'device': device_mem, 'shape': shape})

    def predict(self, input_data):
        # 1. Copy image data to pagelocked host memory
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # 2. Transfer input data to GPU (Host to Device)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. Run Inference on GPU
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. Transfer predictions back to CPU (Device to Host)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        self.stream.synchronize()
        
        # Return a list of output numpy arrays
        return [out['host'].reshape(out['shape']) for out in self.outputs]