import onnx
from onnx import numpy_helper

# 업로드된 ONNX 파일 경로
onnx_file_path = r"D:\gitaddfolder\sources\test\best.onnx"

# ONNX 모델 불러오기
model = onnx.load(onnx_file_path)

# 모델 정보 출력
model_info = onnx.helper.printable_graph(model.graph)

# 모델의 입력과 출력 정보 출력
input_all = [node.name for node in model.graph.input]
input_initializer = [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all) - set(input_initializer))

output_all = [node.name for node in model.graph.output]

(input_all, input_initializer, net_feed_input, output_all, model_info)

if model.metadata_props:
    for prop in model.metadata_props:
        print(f"{prop.key}: {prop.value}")
else:
    print("No metadata found in the model.")