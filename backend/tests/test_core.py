import unittest,tempfile
from pathlib import Path
import numpy as np
from foveamap.grid import addresses,build_grid,evaluate,GridConfig
from foveamap.range_image import range_image
from foveamap.simulation import generate
from foveamap.io import load_scan,write_csv,load_semantickitti,load_rellis_labels,numeric_frame_key,load_kitti_bin_bytes
from foveamap.tracking import Tracker,clusters,extract_objects,TrackingConfig
from foveamap.preprocess import deskew_constant_twist

class CoreTests(unittest.TestCase):
    def test_boundaries(self):
        p=np.array([[r,0,0] for r in [0,9.999999,10,29.999999,30,100,100.001,np.nan]])
        valid,keys,*_=addresses(p)
        self.assertEqual(keys[:,0].tolist(),[0,0,1,1,2,2]);self.assertEqual(valid.sum(),6)
    def test_stats(self):
        g=build_grid(np.array([[10,0,1],[10,0,3]]),[2,2]);c=g['cells'][0]
        self.assertEqual(c['mean'],2);self.assertEqual(c['roughness'],1);self.assertEqual(c['min'],1);self.assertEqual(c['max'],3);self.assertEqual(c['count'],2)
    def test_empty(self):self.assertEqual(build_grid(np.empty((0,3)))['cells'],[])
    def test_conservation(self):
        p,y=generate();g=build_grid(p,y);self.assertEqual(g['metrics']['assigned'],g['metrics']['accepted']);self.assertEqual(g['metrics']['accepted']+g['metrics']['dropped'],len(p))
    def test_config(self):
        with self.assertRaises(ValueError):GridConfig(near_size=0).validate()
    def test_metrics(self):
        e=evaluate([0,0,1],[0,1,1],np.array([[0,0,0],[10,0,0],[30,0,0]]));self.assertEqual(e['iou'][:2],[.5,.5]);self.assertEqual(e['miou'],.5)
        self.assertIsNone(evaluate([-1], [0], np.array([[0,0,0]]))['miou'])
    def test_range_nearest_and_fov(self):
        p=np.array([[2,0,0,.5],[1,0,0,.7],[0,0,10,.8]])
        x,y,r,c,valid=range_image(p,np.array([2,3,1]));self.assertEqual(y[r[0],c[0]],3);self.assertEqual(valid.tolist(),[True,True,False]);self.assertAlmostEqual(x[0,r[0],c[0]],.01)
    def test_io(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.csv';p,y=generate(count=10);write_csv(path,p,y);q,z=load_scan(path);np.testing.assert_allclose(q,p);np.testing.assert_array_equal(y,z)
            label=Path(d)/'test.label';np.array([40,252,999],dtype='<u4').tofile(label);self.assertEqual(load_semantickitti(label,3).tolist(),[0,3,-1])
            rellis=Path(d)/'rellis.label';np.array([8,17+(2<<16)],dtype='<u4').tofile(rellis);r=load_rellis_labels(rellis,2);self.assertEqual(r['semantic'].tolist(),[8,17]);self.assertEqual(r['instance'].tolist(),[0,2])
            with self.assertRaises(ValueError):load_rellis_labels(rellis,3)
            self.assertEqual([p.name for p in sorted([Path('10.bin'),Path('2.bin')],key=numeric_frame_key)],['2.bin','10.bin'])
            with self.assertRaises(ValueError):load_kitti_bin_bytes(b'123')
    def test_tracker(self):
        t=Tracker(config=TrackingConfig(stationary_sensor=True,moving_enter_mps=.5));a=t.update([{'centroid':[0,0,0],'min':[0,0,0],'max':[1,1,1],'dimensions':[1,1,1],'points':8}],0,0);b=t.update([{'centroid':[1,0,0],'min':[1,0,0],'max':[2,1,1],'dimensions':[1,1,1],'points':8}],.5,1);self.assertEqual(a[0]['trackId'],b[0]['trackId']);self.assertEqual(b[0]['velocity'],[2,0,0]);self.assertEqual(b[0]['motionStatus'],'moving')
        c=t.update([{'centroid':[1,0,0],'min':[1,0,0],'max':[2,1,1],'dimensions':[1,1,1],'points':8}],2,5);self.assertEqual(c[0]['trackId'],1)
        u=Tracker().update([{'centroid':[0,0,0],'min':[0,0,0],'max':[1,1,1],'dimensions':[1,1,1],'points':8}],0,0)[0];self.assertEqual(u['motionFrame'],'sensor-relative-uncompensated');self.assertIsNone(u['speedMps'])
    def test_clusters(self):
        p=np.vstack([np.zeros((10,3)),np.ones((10,3))*5]);self.assertEqual(len(clusters(p,np.full(20,3))),2)
    def test_ground_and_extraction(self):
        ground=np.array([[x,y,0] for x in range(3) for y in range(3)]+[[5,0,0],[5.2,0,0]],dtype=float)
        box=np.array([[5+x*.1,0,1+y*.1] for x in range(4) for y in range(4)],dtype=float)
        objects=extract_objects(np.vstack([ground,box]),config=TrackingConfig(min_cluster_points=4))
        self.assertEqual(len(objects),1);self.assertGreater(objects[0]['centroid'][2],1)
    def test_pose_compensated_static_scene(self):
        cfg=TrackingConfig(stationary_sensor=False,moving_enter_mps=.2)
        t=Tracker(config=cfg)
        pose0=np.eye(4);pose1=np.eye(4);pose1[0,3]=1
        a={'centroid':[5,0,0],'min':[4.5,-.5,0],'max':[5.5,.5,1],'dimensions':[1,1,1],'points':20}
        b={'centroid':[4,0,0],'min':[3.5,-.5,0],'max':[4.5,.5,1],'dimensions':[1,1,1],'points':20}
        t.update([a],0,0,pose0);out=t.update([b],1,1,pose1)[0]
        self.assertEqual(out['motionFrame'],'world-compensated');self.assertLess(out['speedMps'],1e-6);self.assertEqual(out['motionStatus'],'stationary')
    def test_deskew(self):
        q=deskew_constant_twist([[1,0,0]],[1],velocity=[2,0,0]);np.testing.assert_allclose(q,[[3,0,0]])

class NeuralTests(unittest.TestCase):
    def test_forward_backward_checkpoint(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch dependency is unavailable')
        from foveamap.model import RangeNet,load_model
        torch.set_num_threads(1)
        model=RangeNet();x=torch.randn(1,5,16,32);out=model(x);self.assertEqual(tuple(out.shape),(1,4,16,32));loss=torch.nn.functional.cross_entropy(out,torch.zeros(1,16,32,dtype=torch.long));loss.backward()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';torch.save({'architecture':'foveamap-range-v1','state_dict':model.state_dict(),'projection':{'height':16,'width':32}},path);loaded,_=load_model(path);loaded.eval();model.eval()
            with torch.no_grad():torch.testing.assert_close(loaded(x),model(x))
if __name__=='__main__':unittest.main()

class DemoModelTests(unittest.TestCase):
    def test_demo_checkpoint_predictions(self):
        from foveamap.point_mlp import predict
        p,y=generate(seed=999,count=100)
        a=predict(p);self.assertEqual(a.shape,(len(p),));self.assertTrue(np.isin(a,[0,1,2,3]).all())
