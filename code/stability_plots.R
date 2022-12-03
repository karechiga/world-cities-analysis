# This script evaluates the varying numbers of clusters

library(ggplot2)
library(tidyverse)


summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  library(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  
  # Rename the "mean" column    
  datac <- rename(datac, c("mean" = measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}


# read in the dataset
files <- list.files(path = "../cluster_data/", pattern = "cities_[0-9]*")
files <- str_sort(files, numeric = TRUE)
stb <- dplyr::as_tibble(read.csv(paste("../cluster_data/", files[1], sep="")))
k = as.numeric(gsub(".*?([0-9]+).*", "\\1", files[1]), "_Cluster",sep="")
stb <- stb %>%
  rename_with(.fn = ~paste0(., "_K=", k, sep=""), 
              .cols = all_of(c("Cluster", "Baseline_Stability", "Stability")))
# Aggregates all stability and cluster data in one DF
for (x in 2:length(files)) {
  data <- read.csv(paste("../cluster_data/", files[x], sep=""))
  k = as.numeric(gsub(".*?([0-9]+).*", "\\1", files[x]), "_Cluster",sep="")
  stb <- stb %>%
    add_column(Cluster=data$Cluster, 
               Baseline_Stability=data$Baseline_Stability,
               Stability=data$Stability) %>%
    rename_with(.fn = ~paste0(., "_K=", k, sep=""), 
                .cols = all_of(c("Cluster", "Baseline_Stability", "Stability")))
}
# Plot average stability vs num_clusters
avg <- select(stb, matches('_K=[0-9]*'), -starts_with("Cluster"))
avg <- select(avg, matches('Stability_K=[0-9]*')) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("type", "num_clusters"),
    names_pattern = '(.*)_K=([0-9]*)',
    values_to = "Stability"
  )
avg$Stability <- avg$Stability * 100
has_na <- unique(avg[is.na(avg$Stability),])
se <- summarySE(avg, measurevar="Stability", groupvars=c("type","num_clusters"), na.rm = T)
se$num_clusters <- as.integer(se$num_clusters)
se <- se[order(se$num_clusters, decreasing = F), ]
se$is_na <- "No"
se[which(se$num_clusters %in% has_na$num_clusters
         & se$type %in% has_na$type),]$is_na <- "Yes"
se$is_na <- factor(se$is_na)
pd <- position_dodge(0.5)
plt1 <- ggplot(se, aes(x=num_clusters,y=Stability, group=type, color=type)) + 
  geom_point(position=pd, aes(shape=is_na), size=3) + 
  scale_shape_manual(values = c(19, 1), name="Undefined Stabilities") +
  geom_errorbar(aes(ymin=Stability-sd, ymax=Stability+sd), width=.5, position=pd) +
  scale_color_discrete(name = "Stability Type", labels = c("Baseline", "With 10% Jackknifing")) +
  scale_y_continuous(breaks=seq(0,100,by=20),limits = c(0,100)) + theme_light() + 
  scale_x_continuous(breaks=seq(min(se$num_clusters),max(se$num_clusters),by=1)) +
  xlab("Number of Clusters") + ylab("Stability (%)") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())


# avg <- select(stb, matches('_K=[0-9]*'), -starts_with("Cluster")) %>% 
#   summarise(across(,mean))
# 
# avg <- select(avg, matches('Stability_K=[0-9]*')) %>%
#   pivot_longer(
#     cols = everything(),
#     names_to = c("type", "num_clusters"),
#     names_pattern = '(.*)_K=([0-9]*)',
#     values_to = "Stability"
#   )
# 
# avg$num_clusters <- as.integer(avg$num_clusters)
# avg <- avg[order(avg$num_clusters, decreasing = F), ]
# 
# plt1 <- ggplot(avg, aes(x=num_clusters,y=Stability*100, group=type, color=type)) + 
#   geom_line() + geom_point() + geom_errorbar(aes(ymin=len-ci, ymax=len+ci), width=.1, position=pd)
#   + xlab("Number of Clusters") + ylab("Mean Stability (%)") +
#   labs(title="Stability of K-Means Clustered Cities Versus the Number of Clusters") +
#   scale_color_discrete(name = "Stability Type", labels = c("Baseline", "With 10% Jackknifing")) +
#   scale_y_continuous(breaks=seq(0,100,by=20),limits = c(0,100)) + theme_light() + 
#   scale_x_continuous(breaks=seq(min(avg$num_clusters),max(avg$num_clusters),by=2))

show(plt1)

bp <- stb %>%
  select(matches('_K=[0-9]*'), -starts_with("Cluster"), -starts_with("Baseline")) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("num_clusters"),
    names_prefix = 'Stability_K=',
    values_to = "Stability"
  )
bp$num_clusters <- as.integer(bp$num_clusters)
bp <- bp[order(bp$num_clusters, decreasing = F), ]

# Creates a boxplot
plt2 <- ggplot(bp, aes(x=factor(num_clusters),y=Stability*100, color=factor(num_clusters))) + 
  geom_boxplot(outlier.alpha = 0.04, outlier.size = 1) + xlab("Number of Clusters") + 
  ylab("Stability With Jackknifing (%)") + 
  labs(title="Stability of K-Means Clustered Cities vs the Number of Clusters") +
  scale_y_continuous(breaks=seq(0,100,by=20),limits = c(0,100)) + theme_classic() + theme(legend.position = "none")

# Plot the Average Stability per Cluster
per_cluster <- select(stb, matches('_K=[0-9]*'))
num_clusters <- unique(as.numeric(gsub(".*?([0-9]+).*", "\\1", names(per_cluster)), "_K=",sep=""))
avg_per <- data.frame(num_clusters = integer(0), type = character(0), 
                      N=integer(0),Stability = numeric(0),sd = numeric(0),is_na = character(0))
for (i in num_clusters) {
  temp <- select(per_cluster, ends_with(paste('_K=', i,sep="")))
  temp <- aggregate(temp[,2:3], temp[,1], FUN=mean)
  temp <- select(temp, matches('Stability_K=[0-9]*')) %>%
    pivot_longer(
      cols = everything(),
      names_to = c("type", "num_clusters"),
      names_pattern = '(.*)_K=([0-9]*)',
      values_to = "Stability"
    )
  temp$Stability <- temp$Stability * 100
  temp <- summarySE(temp, measurevar="Stability", groupvars=c("type"), na.rm = T)
  avg_per[nrow(avg_per) + 1,] = c(i, "Baseline_per_cluster", temp[1,2], temp[1,3],
                                  temp[1,4], ifelse(temp[1,2] < i, "Yes", "No"))
  avg_per[nrow(avg_per) + 1,] = c(i, "Jackknifing_per_cluster", temp[2,2], temp[2,3],
                                  temp[2,4], ifelse(temp[2,2] < i, "Yes", "No"))
  
}
avg_per$num_clusters <- as.integer(avg_per$num_clusters)
avg_per$Stability <- as.numeric(avg_per$Stability)
avg_per$sd <- as.numeric(avg_per$sd)
avg_per$is_na <- factor(se$is_na)
pd <- position_dodge(0.5)

plt3 <- ggplot(avg_per, aes(x=num_clusters,y=Stability, group=type, color=type)) + 
  geom_point(position=pd, aes(shape=is_na), size=3) + 
  scale_shape_manual(values = c(19, 1), name="Undefined Stabilities") +
  geom_errorbar(aes(ymin=Stability-sd, ymax=Stability+sd), width=.5, position=pd) +
  scale_color_discrete(name = "Stability Type", labels = c("Baseline", "With 10% Jackknifing")) +
  scale_y_continuous(breaks=seq(0,100,by=20),limits = c(0,101)) + theme_light() + 
  scale_x_continuous(breaks=seq(min(se$num_clusters),max(se$num_clusters),by=1)) +
  xlab("Number of Clusters") + ylab("Stability (%)") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# saving the plots as pngs in the "figures" folder 
ggsave("stability_per_city_points.png", plt1, path = "../figures")
ggsave("num_clusters_stability_box.png", plt2, path = "../figures")
ggsave("stability_per_cluster_points.png", plt3, path = "../figures")

### Individual number of clusters stability plot

# read in the dataset
data <- read.csv("../cluster_data/cities_6.csv")
data$Cluster <- factor(data$Cluster)

# Creates a boxplot
plt4 <- ggplot(data, aes(x=Cluster,group=Cluster,y = Stability*100, color=Cluster)) + 
  geom_boxplot(outlier.shape = NA) + geom_jitter(alpha=0.17, width=0.3) +
  xlab("Cluster") + ylab("Stability with Jackknifing (%)") +
  labs(title=paste("Stability of K-Means Clustered Cities Boxplots (k =", length(unique(data$Cluster)), ")"),) +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 12), plot.title = element_text(size = 15) ) +
  scale_y_continuous(breaks=seq(0,100,by=20),limits = c(0,100)) + theme_light() + theme(legend.position = "none")

# saving the plots as pngs in the "figures" folder 
ggsave(paste("Cluster_Stability_", length(unique(data$Cluster)) ,".png", sep=""), plt4, path = "../figures")