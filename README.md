# world-cities-analysis

<!-- ABOUT THE PROJECT -->


### Built With
Python 3.9.12, GeoPandas 0.9, GDAL 3.02, scikit-learn 1.1.1

### Datasets
We use the 6,018 city boundaries [defined in a .shp file](https://geo.nyu.edu/catalog/stanford-yk247bg4748) and analyzed the cities based on the following features:

* [WorldClim bioclimatic data](https://worldclim.org/data/worldclim21.html): Used the 19 bioclimatic .tif files at a resolution of 5 arcminutes.
* [Landscan population data](https://landscan.ornl.gov/): Used population data from the years 2000 and 2021 in 30 arcescond resolution .tif format.
* [Artificial Sky Brightness](https://doi.org/10.5880/GFZ.1.4.2016.001): Brightness data from 2015 extracted from a 30 arcsecond .tif file.
* [Road Density](https://www.globio.info/download-grip-dataset): Extracted from a 5 arcminute resolution .asc file.
* [Human Modification](https://figshare.com/articles/dataset/Global_Human_Modification/7283087): This dataset requires a the .tif to be reprojected to EPSG:4326. This can be achieved using GDAL in the command line:
```
gdalwarp -t_srs EPSG:4326 '../../datasets/human_modification/gHM.tif' '../../datasets/human_modification/gHM_mod.tif'
```
The original "gHM.tif" file should be moved to another unused folder or deleted.
* [Land Usage](https://www.pbl.nl/en/image/links/hyde): We used the cropland and grazing land data from 2017. Files are 5 arcminute .asc format.
* [Elevation](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-multi-resolution-terrain-elevation): Originally exported as a .shp file, then it should be rasterized as a new .tif file using GDAL in a command line:
```
gdal_rasterize -a MEAN_ELEV "../../datasets/elevation/GMTED2010_Spatial_Metadata.shp" -tr 0.008333333333333 0.008333333333333 "../../datasets/elevation/elevation.tif"
```

Unused but also available to extract 
* [Paleoclim](http://www.paleoclim.org/)
* [Urban Heat](https://figshare.com/articles/dataset/Global_Urban_Heat_Island_Intensification/7897433)

Run the following to see all available options to extract data using the city_data.py script:
```
python city_data.py -h   
```
```
Usage: city_data.py [options]

Options:
  -h, --help          show this help message and exit
  -c C, --plotcity=C  City to be plotted (case sensitive; options are any city
                      in our database (Boston, Los Angeles, Tokyo, Beijing,
                      etc.)
  -w, --worldclim     Extract WorldClim data stored at
                      "../../datasets/worldclim"
  -p, --paleoclim     Extract PaleoClim data stored at
                      "../../datasets/paleoclim"
  -l, --landscan      Extract Landscan data stored at
                      "../../datasets/landscan"
  -b, --brightness    Extract Sky Brightness data stored at
                      "../../datasets/brightness"
  -r, --roads         Extract road density data stored at
                      "../../datasets/roads"
  -m, --human         Extract human modification data stored at
                      "../../datasets/human_modification"
  -u, --urban_heat    Extract urban heat data stored at "../../datasets/roads"
  -y, --land_use      Extract land use data stored at
                      "../../datasets/land_use"
  -e, --elevation     Extract elevation data stored at
                      "../../datasets/elevation"
  -g, --geodist       Output geographical distances between cities.
```




### City Features
