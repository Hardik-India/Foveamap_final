import numpy as np

def generate(frame=0, seed=42, count=12000):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-70,70,count), rng.uniform(-45,45,count)
    ground = np.abs(y) < 7
    z = np.where(ground,0,.23)+rng.normal(0,.015,count)
    labels = (~ground).astype(np.int64)
    pothole = (x-7)**2+(y-1)**2 < 4
    z[pothole] = -.4+rng.normal(0,.03,pothole.sum()); labels[pothole] = 1
    points = [np.column_stack([x,y,z,rng.uniform(.1,1,count)])]
    truths = [labels]
    def box(cx,cy,w,d,h,label,n):
        xyz=rng.uniform(-.5,.5,(n,3))*[w,d,h]+[cx,cy,h/2]
        xyz[::3,0]=cx+w/2; xyz[1::3,1]=cy+d/2; xyz[2::3,2]=h
        points.append(np.column_stack([xyz,rng.uniform(.1,1,n)])); truths.append(np.full(n,label))
    for cx in range(-50,60,15):
        box(cx,15,9,7,7,2,250); box(cx,-18,10,9,9,2,250)
    box(-15+(frame*.45)%90,3,4.5,1.9,1.6,3,300)
    box(32-(frame*.25)%80,-3.5,4,1.8,1.5,3,300)
    box(8,6-np.sin(frame*.035)*3,.5,.5,1.7,3,80)
    return np.vstack(points).astype(np.float32), np.concatenate(truths)
