"""Small independent ONNX fixture and negative converter tests; no ViT names or weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
spec=importlib.util.spec_from_file_location('converter',Path(__file__).resolve().parents[1]/'tools/convert.py')
converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)


def fixture():
    weights={'w':np.arange(24,dtype=np.float32).reshape(3,2,2,2)/30-0.3,
        'bias':np.array([0.2,-0.1,0.3],np.float32),'shape':np.array([1,4,3],np.int64),
        'scale':np.array([0.7,1.1,0.9],np.float32),'offset':np.array([0.1,0.2,-0.3],np.float32),
        'projection':np.arange(12,dtype=np.float32).reshape(3,4)/13,
        'index':np.array(-1,np.int64),'head':np.arange(8,dtype=np.float32).reshape(2,4)/9,
        'head_bias':np.array([0.1,-0.2],np.float32)}
    nodes=[h.make_node('Conv',['x','w','bias'],['conv'],kernel_shape=[2,2],strides=[2,2]),
        h.make_node('Transpose',['conv'],['transposed'],perm=[0,2,3,1]),
        h.make_node('Reshape',['transposed','shape'],['reshaped']),
        h.make_node('LayerNormalization',['reshaped','scale','offset'],['norm'],axis=-1,epsilon=0.01),
        h.make_node('Erf',['norm'],['erf']),h.make_node('MatMul',['erf','projection'],['projected']),
        h.make_node('Softmax',['projected'],['probabilities'],axis=-1),
        h.make_node('Gather',['probabilities','index'],['selected'],axis=1),
        h.make_node('Gemm',['selected','head','head_bias'],['result'],alpha=0.5,beta=2.0,transB=1)]
    shapes={'conv':[1,3,2,2],'norm':[1,4,3],'erf':[1,4,3],'selected':[1,4],'result':[1,2]}
    graph=h.make_graph(nodes,'synthetic',[h.make_tensor_value_info('x',T.FLOAT,[1,2,4,4])],
        [h.make_tensor_value_info(n,T.FLOAT,s) for n,s in shapes.items()],
        [nh.from_array(v,n) for n,v in weights.items()])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=10
    return model

def spatial_fixture():
    data=np.linspace(-1,1,240,dtype=np.float32).reshape(2,4,5,6)
    weights={
        'grouped_w':np.sin(np.arange(72,dtype=np.float32)).reshape(6,2,2,3)/7,
        'grouped_b':np.linspace(-0.2,0.2,6,dtype=np.float32),
        'depth_w':np.cos(np.arange(24,dtype=np.float32)).reshape(4,1,3,2)/5,
        'point_w':np.arange(12,dtype=np.float32).reshape(3,4,1,1)/17,
        'overlap_w':np.sin(np.arange(72,dtype=np.float32)).reshape(2,4,3,3)/11,
        'pads':np.array([0,0,1,2,0,0,0,1],np.int64),'fill':np.array(0,np.float32),
        'lo':np.array(-0.4,np.float32),'hi':np.array(0.7,np.float32)}
    nodes=[
        h.make_node('Conv',['x','grouped_w','grouped_b'],['grouped'],group=2,kernel_shape=[2,3],strides=[2,1],dilations=[2,1],pads=[1,0,0,2]),
        h.make_node('Conv',['x','depth_w'],['depthwise'],group=4,kernel_shape=[3,2],strides=[2,1],dilations=[1,2],pads=[0,1,2,0]),
        h.make_node('Conv',['x','point_w'],['pointwise'],kernel_shape=[1,1]),
        h.make_node('Conv',['x','overlap_w'],['overlap'],kernel_shape=[3,3],strides=[2,2],pads=[1,1,1,1]),
        h.make_node('Pad',['x','pads','fill'],['padded'],mode='constant'),
        h.make_node('Clip',['padded','lo','hi'],['clipped']),
        h.make_node('GlobalAveragePool',['clipped'],['pooled']),
        h.make_node('Flatten',['pooled'],['flat'],axis=1)]
    shapes={'grouped':[2,6,2,6],'depthwise':[2,4,3,5],'pointwise':[2,3,5,6],
        'overlap':[2,2,3,3],'padded':[2,4,6,9],'clipped':[2,4,6,9],'pooled':[2,4,1,1],'flat':[2,4]}
    graph=h.make_graph(nodes,'spatial',[h.make_tensor_value_info('x',T.FLOAT,list(data.shape))],
        [h.make_tensor_value_info(n,T.FLOAT,s) for n,s in shapes.items()],[nh.from_array(v,n) for n,v in weights.items()])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=10
    return model,data

class Conversion(unittest.TestCase):
    def check_rejected(self,model):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.onnx';onnx.save(model,path)
            with self.assertRaises((ValueError,onnx.checker.ValidationError,onnx.shape_inference.InferenceError)):
                converter.convert(path,Path(tmp)/'converted')
    def test_unsqueeze_lowering(self):
        for axes in [[0, -1], [1]]:
            rank=2+len(axes);normalized=[a%rank for a in axes];source_shape=iter([2,3])
            shape=[1 if i in normalized else next(source_shape) for i in range(rank)]
            graph=h.make_graph([h.make_node('Unsqueeze',['x','axes'],['y'])], 'unsqueeze',
                [h.make_tensor_value_info('x',T.FLOAT,[2,3])],[h.make_tensor_value_info('y',T.FLOAT,shape)],
                [nh.from_array(np.array(axes,np.int64),'axes')])
            model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=10
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'input.onnx';onnx.save(model,path)
                manifest=converter.convert(path,Path(tmp)/'converted')
                node=manifest['nodes'][0]
                self.assertEqual(node['op'],'Reshape')
                data=np.arange(6,dtype=np.float32).reshape(2,3)
                expected=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']).run(None,{'x':data})[0]
                np.testing.assert_array_equal(data.reshape(node['attrs']['shape']),expected)
            graph.initializer[0].CopyFrom(nh.from_array(np.array([0,0],np.int64),'axes'))
            self.check_rejected(h.make_model(graph,opset_imports=[h.make_opsetid('',17)]))
    def test_conv1d_slice_fixture(self):
        data=np.linspace(-1,1,72,dtype=np.float32).reshape(2,4,9)
        weights={'w':np.cos(np.arange(36,dtype=np.float32)).reshape(6,2,3)/7,
            'start':np.array([-3],np.int64),'end':np.array([-1],np.int64),'axes':np.array([-1],np.int64),'steps':np.array([1],np.int64)}
        nodes=[h.make_node('Conv',['x','w'],['conv'],group=2,strides=[2],dilations=[2],pads=[1,2]),
            h.make_node('Slice',['conv','start','end','axes','steps'],['sliced'])]
        graph=h.make_graph(nodes,'conv1d-slice',[h.make_tensor_value_info('x',T.FLOAT,[2,4,9])],
            [h.make_tensor_value_info('conv',T.FLOAT,[2,6,4]),h.make_tensor_value_info('sliced',T.FLOAT,[2,6,2])],
            [nh.from_array(v,n) for n,v in weights.items()])
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=10
        out=Path(__file__).with_name('fixtures')/'conv1d';out.mkdir(exist_ok=True)
        path=out/'synthetic.onnx';onnx.save(model,path);converter.convert(path,out)
        values=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']).run(None,{'x':data})
        (out/'reference.json').write_text(json.dumps({'input':data.tolist(),'expected':dict(zip(['conv','sliced'],[v.tolist() for v in values]))}))
        graph.initializer[-1].CopyFrom(nh.from_array(np.array([2],np.int64),'steps'))
        self.check_rejected(h.make_model(graph,opset_imports=[h.make_opsetid('',17)]))
    def test_supported_fixture(self):
        model=fixture();out=Path(__file__).with_name('fixtures');out.mkdir(exist_ok=True)
        path=out/'synthetic.onnx';onnx.save(model,path)
        converter.convert(path,out)
        data=np.linspace(-1,1,32,dtype=np.float32).reshape(1,2,4,4)
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        expected=dict(zip([o.name for o in session.get_outputs()],[v.tolist() for v in session.run(None,{'x':data})]))
        (out/'reference.json').write_text(json.dumps({'input':data.tolist(),'expected':expected},indent=2)+'\n')
    def test_reject_opset(self):
        model=fixture();model.opset_import[0].version=18;self.check_rejected(model)
    def test_reject_dynamic_input(self):
        model=fixture();dim=model.graph.input[0].type.tensor_type.shape.dim[0];dim.ClearField('dim_value');dim.dim_param='batch';self.check_rejected(model)
    def test_reject_implicit_auto_padding(self):
        model=fixture();model.graph.node[0].attribute.append(h.make_attribute('auto_pad','SAME_UPPER'))
        self.check_rejected(model)
    def test_spatial_fixture(self):
        model,data=spatial_fixture();out=Path(__file__).with_name('fixtures')/'spatial';out.mkdir(exist_ok=True)
        path=out/'synthetic.onnx';onnx.save(model,path);converter.convert(path,out)
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        expected=dict(zip([o.name for o in session.get_outputs()],[v.tolist() for v in session.run(None,{'x':data})]))
        (out/'reference.json').write_text(json.dumps({'input':data.tolist(),'expected':expected},indent=2)+'\n')
    def test_reject_nonzero_padding_fill(self):
        model,_=spatial_fixture()
        for i,t in enumerate(model.graph.initializer):
            if t.name=='fill':model.graph.initializer[i].CopyFrom(nh.from_array(np.array(1,dtype=np.float32),'fill'))
        self.check_rejected(model)
    def test_reject_dynamic_conv_bias(self):
        model=fixture()
        for i,t in enumerate(model.graph.initializer):
            if t.name=='bias':del model.graph.initializer[i];break
        model.graph.input.append(h.make_tensor_value_info('bias',T.FLOAT,[3]))
        self.check_rejected(model)
    def test_reject_unsupported_runtime_op(self):
        model=fixture();model.graph.node[4].op_type='Sigmoid';self.check_rejected(model)

if __name__=='__main__':unittest.main()
