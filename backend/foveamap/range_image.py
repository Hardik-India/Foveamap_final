import numpy as np

def range_image(points, labels=None, height=64, width=512, fov_up=15., fov_down=-25.):
    """Nearest-return z-buffer; out-of-FOV points have invalid mask, never clip into edge rows."""
    p=np.asarray(points)
    if not height>0 or not width>0 or not fov_up>fov_down:
        raise ValueError('Invalid range projection dimensions/FOV')
    depth=np.linalg.norm(p[:,:3],axis=1)
    valid=np.isfinite(p[:,:4]).all(axis=1)&(depth>1e-6)
    pitch=np.arcsin(np.clip(p[:,2]/np.maximum(depth,1e-6),-1,1))
    lo,hi=np.deg2rad([fov_down,fov_up])
    valid &= (pitch>=lo)&(pitch<=hi)
    col=np.floor(np.mod(np.arctan2(p[:,1],p[:,0])+np.pi,2*np.pi)/(2*np.pi)*width).astype(int)
    row=np.clip(np.floor((hi-pitch)/(hi-lo)*height).astype(int),0,height-1)
    features=np.zeros((5,height,width),dtype=np.float32)
    target=np.full((height,width),-1,dtype=np.int64)
    # Stable far-to-near overwrite leaves the closest return at each pixel.
    ids=np.flatnonzero(valid); ids=ids[np.argsort(-depth[ids],kind='stable')]
    for i in ids:
        features[:,row[i],col[i]]=[depth[i]/100,p[i,0]/100,p[i,1]/100,p[i,2]/10,p[i,3]]
        if labels is not None: target[row[i],col[i]]=int(labels[i])
    return features,target,row,col,valid
