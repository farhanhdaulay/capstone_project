import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import numpy as np

class TRTWrapper:
    def __init__(self, engine_path):
        # 1. Capture the CUDA context created by autoinit
        self.ctx = pycuda.autoinit.context
        
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")
        
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            if -1 in shape:
                shape = self.context.get_tensor_shape(name)
                
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            self.context.set_tensor_address(name, int(device_mem))
            
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            binding = {
                "name": name,
                "host": cuda.pagelocked_empty(trt.volume(shape), dtype),
                "device": device_mem,
                "shape": shape,
                "dtype": dtype
            }
            
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def predict(self, input_data):
        # 2. PUSH the CUDA context into the current worker thread
        self.ctx.push()
        try:
            input_data = np.ascontiguousarray(input_data)
            
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
                
            self.stream.synchronize()
            
            return [out['host'].reshape(out['shape']) for out in self.outputs]
        finally:
            # 3. POP the context so we don't leak memory or block other threads
            self.ctx.pop()