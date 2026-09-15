import numpy as np

def deskew_constant_twist(points, timestamps, velocity=(0.,0.,0.), yaw_rate=0.):
    """Transform returns into the scan-start frame under a supplied constant body twist.
    timestamps: seconds from scan start; yaw_rate: rad/s; velocity: m/s in initial frame.
    This is an explicit approximation, not IMU / odometry fusion.
    """
    p=np.array(points,dtype=float,copy=True); t=np.asarray(timestamps)
    if t.shape != (len(p),) or not np.isfinite(t).all() or np.any(t<0): raise ValueError('Invalid per-point timestamps')
    theta=t*yaw_rate;c,s=np.cos(theta),np.sin(theta)
    x,y=p[:,0].copy(),p[:,1].copy();p[:,0]=c*x-s*y;p[:,1]=s*x+c*y
    p[:,:3]+=t[:,None]*np.asarray(velocity)
    return p

def voxel_downsample(points, size=.02):
    """Keep a deterministic representative per Cartesian voxel (before classification)."""
    if size<=0: raise ValueError('Voxel size must be positive')
    p=np.asarray(points); _,indices=np.unique(np.floor(p[:,:3]/size).astype(np.int64),axis=0,return_index=True)
    indices=np.sort(indices)
    return p[indices],indices
