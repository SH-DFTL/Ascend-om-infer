import acl
import numpy as np
  
class ACL_inference(object):
    def __init__(self, device_id):
        self.device_id = device_id
        acl.init()
        acl.rt.set_device(self.device_id)
        self.context, _ = acl.rt.create_context(self.device_id)
        self.ACL_MEMCPY_HOST_TO_DEVICE = 1
        self.ACL_MEMCPY_DEVICE_TO_HOST = 2
        self.ACL_MEM_MALLOC_HUGE_ONLY = 2
        self.model_id = None
        self.model_desc = None
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.input_data = []
        self.output_data = []
 
    def init(self, model_path):
        self.model_id, _ = acl.mdl.load_from_file(model_path)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        self.gen_input_dataset()
        self.gen_output_dataset()
 
    def gen_output_dataset(self):
        self.load_output_dataset = acl.mdl.create_dataset()
        # 获取模型输出的数量。
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        # 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中。
        for i in range(output_size):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            # 申请输出内存。
            buffer, ret = acl.rt.malloc(buffer_size, self.ACL_MEM_MALLOC_HUGE_ONLY)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_output_dataset, data)
            self.output_data.append({"buffer": buffer, "size": buffer_size})
 
    def gen_input_dataset(self):
        self.load_input_dataset = acl.mdl.create_dataset()
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        print("input_size:",input_size)
        for i in range(1):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, self.ACL_MEM_MALLOC_HUGE_ONLY)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_input_dataset, data)
            self.input_data.append({"buffer": buffer, "size": buffer_size})
            print("add to self.input_data, buffer",buffer,"size:",buffer_size)
 
    def process_output(self):
        inference_result = []
        for i, item in enumerate(self.output_data):
            dims = acl.mdl.get_output_dims(self.model_desc, i)
            shape = tuple(dims[i]["dims"])
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # 将推理输出数据从Device传输到Host。
            acl.rt.memcpy(buffer_host, self.output_data[i]["size"], self.output_data[i]["buffer"],
                          self.output_data[i]["size"], self.ACL_MEMCPY_DEVICE_TO_HOST)
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            data = np.frombuffer(bytes_out, dtype=np.float32).reshape(shape)
            inference_result.append(data)
        return inference_result
 
    def load_input_data(self, img):
        bytes_data = img.tobytes()
        np_ptr = acl.util.bytes_to_ptr(bytes_data)
        # 将图片数据从Host传输到Device。
        acl.rt.memcpy(self.input_data[0]["buffer"], self.input_data[0]["size"], np_ptr,
                      self.input_data[0]["size"], self.ACL_MEMCPY_HOST_TO_DEVICE)
 
    def execute(self):
        acl.mdl.execute(self.model_id, self.load_input_dataset, self.load_output_dataset)
 
    def destory(self):
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
