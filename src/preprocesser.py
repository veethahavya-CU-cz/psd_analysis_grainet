import os, sys
import numpy as np
import fiona
from osgeo import gdal



### Core function definitions

def hisGen(shp_file, shp_dir):
    shape_bound = fiona.open(shp_file)
    items = list(shape_bound)
    gr_sz = []
    for i in range(len(items)):
        if items[i]["properties"]["width"] < 0.1:
            gr_sz.append(max(items[i]["properties"]["height"], items[i]["properties"]["width"]))
        else:
            gr_sz.append(items[i]["properties"]["width"])
    gr_sz = np.array(gr_sz)
    dm = gr_sz.mean()*100

    bns = np.array([0,1,2,3,4,6,8,10,12,15,20,25,30,35,40,50,60,80,100,120,150,200,250])*0.01
    his = np.array(np.histogram(gr_sz, bins=bns)[0])
    his = np.cumsum(his/his.sum())
    return his, dm
    
def getRGB(tif_file):
    ds = gdal.Open(tif_file)
    rgb = np.stack((ds.GetRasterBand(b).ReadAsArray() for b in (1,2,3)), axis=2)
    return rgb



### Iterative execution of functions

# define the paths: wdir - working dir, shp_dir - dir with all shapefiles, and tif_dir - dir with all tif files    
wdir = os.path.join('data', 'temp_preprocessor') #orthoimages aka orthomosaics
if not os.path.exists(wdir):
    os.makedirs(wdir)
npz_dir = os.path.join('data')
shp_dir = os.path.join('data', 'annotations')
tif_dir = os.path.join('data', 'orthomoasic_tiles')

# dm and his calculation
os.chdir(shp_dir)
his_list, dm_list, name_shp = [], [], []
for file in os.listdir(shp_dir):
    if file.endswith(".shp"):
        his_part, dm_part = hisGen(file, shp_dir)
        his_list.append(his_part)
        dm_list.append(dm_part)
        name_shp.append(os.path.splitext(file)[0])
his = np.stack(his_list)
dm = np.array(dm_list)

# rgb extraction from images
os.chdir(tif_dir)
img_list, name_tif = [], []
for file in os.listdir(tif_dir):
    if file.endswith(".tif"):
        img_list.append(getRGB(file))
        name_tif.append(os.path.splitext(file)[0])
img = np.stack(img_list)
        
# safety check
if len(name_shp) != len(name_tif):
    sys.exit("Unequal number of tifs and shps. Aborting npz packaging!")
names = np.array(name_tif, dtype='O')

# write the npz output
os.chdir(npz_dir)
np.savez("preprocessed_data", images=img, histograms=his, tile_names=names, dm=dm)