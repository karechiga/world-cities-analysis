# world-cities-analysis

<!-- ABOUT THE PROJECT -->


### Built With
Python 3.9.12, GeoPandas 0.9, GDAL 3.02, scikit-learn 1.1.1

### Datasets
We use the 6,018 city boundaries [defined in a .shp file](https://geo.nyu.edu/catalog/stanford-yk247bg4748) (Export the file in EPSG:4326). After downloading, move the extracted files into the directory ``` "../../city-boundaries/" ``` relative to the current code folder.

Continents for each city were determined using [another shapefile](https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-physical-labels/) (download the "label areas"). Move the extracted data to ``` "../../continent-boundaries/" ``` relative to the current code folder.

Cities were analyzed based on the following extracted datasets. 

Note that all of the extracted CSV data for our analyses has been pushed to the "csv_data" folder in this repository. It is recommended to use those files instead of re-extracting the data as explained below:

* [WorldClim bioclimatic data](https://worldclim.org/data/worldclim21.html): Used the 19 bioclimatic .tif files at a resolution of 5 arcminutes.
* [Landscan population data](https://landscan.ornl.gov/): Used population data from the years 2000 and 2021 in 30 arcescond resolution .tif format.
* [Artificial Sky Brightness](https://doi.org/10.5880/GFZ.1.4.2016.001): Brightness data from 2015 extracted from a 30 arcsecond .tif file.
* [Road Density](https://www.globio.info/download-grip-dataset): Extracted from a 5 arcminute resolution .asc file.
* [Human Modification](https://figshare.com/articles/dataset/Global_Human_Modification/7283087): This dataset requires a the .tif to be reprojected to EPSG:4326. This can be achieved using GDAL in the command line:
```
> gdalwarp -t_srs EPSG:4326 '../../datasets/human_modification/gHM.tif' '../../datasets/human_modification/gHM_mod.tif'
```
The original "gHM.tif" file should be moved to another unused folder or deleted.
* [Land Usage](https://www.pbl.nl/en/image/links/hyde): We used the cropland and grazing land data from 2017. Files are 5 arcminute .asc format.
* [Elevation](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-multi-resolution-terrain-elevation): Originally exported as a .shp file, then it should be rasterized as a new .tif file using GDAL in a command line:
```
> gdal_rasterize -a MEAN_ELEV "../../datasets/elevation/GMTED2010_Spatial_Metadata.shp" -tr 0.008333333333333 0.008333333333333 "../../datasets/elevation/elevation.tif"
```

Unused but also available to extract 
* [Paleoclim](http://www.paleoclim.org/)
* [Urban Heat](https://figshare.com/articles/dataset/Global_Urban_Heat_Island_Intensification/7897433)

Run the following to see all available options to extract data using the city_data.py script:
```
> python city_data.py -h   
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

### City Analysis
Use the cities.py script with the following options to perform analyses on the data. Example output plots are located in the "figures" folder.
```
> python cities.py -h 
```

```
Usage: cities.py [options]

Options:
  -h, --help            show this help message and exit
  -s, --stability       Clusters the cities based on features located at
                        "../csv_data/". Then calculates the cluster stability
                        by running several iterations of clustering.
  -c, --cluster         Clusters city data located at "../csv_data/", then
                        plots the clusters in a geographic representation.
  -t, --centers_to_csv  Saves calculated cluster centroids to CSV file (use
                        with python cities.py -c or -p).
  -p, --pca             Clusters then plots the cities with their first two
                        Principle Components
  -e, --elbow           Uses centroid CSV files for different k values, and
                        Euclidean calculates distances between cities and
                        clusters. Plots an Elbow plot and generates a csv for
                        cluster distances and city distances from each other.
  -m METHOD, --method=METHOD
                        Clustering method to perform. Type either "dbscan" or
                        "kmeans". (Default: kmeans)
  -k K, --num_clusters=K
                        Number of clusters to partition the data in k-means
                        clustering. Default: 6
  -i C, --cluster_iters=C
                        Number of clustering iterations to calculate baseline
                        clusters. Default: 10
  -b B, --baseline_stability_iters=B
                        Number of iterations to calculate the stability of the
                        baseline clusters. Default: 100
  -n N, --stability_iters=N
                        Number of iterations to calculate the cluster
                        stability for each city. Default: 100
  -f F, --drop_features=F
                        Percentage of features to drop at each iteration of
                        stability calculations. Default: 10
  -d D, --drop_rows=D   Percentage of rows to drop at each iteration of
                        stability calculations. Default: 10
  -r R, --random_seed=R
                        Random seed initialization. Default: 10
```
### SLURM usage

It is recommended to use an HPC and schedule tasks using a SLURM script. It can take several hours for many of the datasets to finish extracting the data to CSV.
Example SLURM script to extract the Landscan data:
```
#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=get_landscan_data
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out
### REQUIRED. Specify the PI group for this job
#SBATCH --account=[enter your group here]
### Optional. Request email when job begins and ends
#SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
#SBATCH --mail-user=[your email address]
### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=4
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=5gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=24:00:00


# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Install necessary modules
source ~/.bashrc && conda activate cities
### change to your script’s directory
cd ~/world-cities-analysis/code
### Run your work
echo "Extracting Landscan population data..."
python city_data.py -l
sleep 10
```

Example SLURM for calculating cluster stability:
```
#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=cluster_stability
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out
### REQUIRED. Specify the PI group for this job
#SBATCH --account=[your group name]
### Optional. Request email when job begins and ends
#SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
#SBATCH --mail-user=[your email]
### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=4
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=5gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=24:00:00


# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Install necessary modules
source ~/.bashrc && conda activate cities
### change to your script’s directory
cd ~/world-cities-analysis/code
### Run your work
echo "Analyzing k=2 cluster stabilities..."
python cities.py -s -k 2 -b 1000 -i 100 -n 10000

echo "Analyzing k=3 cluster stabilities..."
python cities.py -s -k 3 -b 1000 -i 100 -n 10000

echo "Analyzing k=4 cluster stabilities..."
python cities.py -s -k 4 -b 1000 -i 100 -n 10000

echo "Analyzing k=5 cluster stabilities..."
python cities.py -s -k 5 -b 1000 -i 100 -n 10000

echo "Analyzing k=6 cluster stabilities..."
python cities.py -s -k 6 -b 1000 -i 100 -n 10000

sleep 10
```

### Authors

Kyle Arechiga - kylearechiga@arizona.edu

Cristian Roman Palacios - cromanpa94@arizona.edu
