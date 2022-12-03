# world-cities-analysis

<!-- ABOUT THE PROJECT -->


### Built With
Python 3.9.12, GeoPandas 0.9, GDAL 3.02, scikit-learn 1.1.1

### Datasets

To rasterize the elevation .shp file:
```
gdal_rasterize -a MEAN_ELEV "../../datasets/elevation/GMTED2010_Spatial_Metadata.shp" -tr 0.008333333333333 0.008333333333333 "../../datasets/elevation/elevation.tif"
```

To change the projection of the human modification to EPSG:4326
```
gdalwarp -t_srs EPSG:4326 '../../datasets/human_modification/gHM.tif' '../../datasets/human_modification/gHM_mod.tif'
```

### City Features
