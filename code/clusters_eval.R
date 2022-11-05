# This script evaluates the varying numbers of clusters

library(ggplot2)
library(tidyverse)

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
avg <- select(stb, matches('_K=[0-9]*'), -starts_with("Cluster")) %>% 
  summarise(across(,mean))

# Plot average stability per cluster vs num_clusters
temp <- select(stb, matches('_K=[0-9]*'))
num_clusters <- unique(as.numeric(gsub(".*?([0-9]+).*", "\\1", names(temp)), "_K=",sep=""))
per_cluster <- data.frame()


avg <- select(avg, matches('Stability_K=[0-9]*')) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("type", "num_clusters"),
    names_pattern = '(.*)_K=([0-9]*)',
    values_to = "Stability"
  )

avg$num_clusters <- as.integer(avg$num_clusters)
avg <- avg[order(avg$num_clusters, decreasing = F), ]

plt1 <- ggplot(avg, aes(x=num_clusters,y=Stability*100, group=type, color=type)) + 
  geom_line() + geom_point() + xlab("Number of Clusters") + ylab("Mean Stability (%)") +
  labs(title="Stability of K-Means Clustered Cities Versus the Number of Clusters") +
  scale_color_discrete(name = "Stability Type", labels = c("Baseline", "With 10% Jackknifing")) +
  scale_y_continuous(breaks=seq(0,100,by=20),limits = c(0,100)) + theme_light() + 
  scale_x_continuous(breaks=seq(min(avg$num_clusters),max(avg$num_clusters),by=2))

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

# saving the plots as pngs in the "figures" folder 
ggsave("num_clusters_stability_line.png", plt1, path = "../figures")
ggsave("num_clusters_stability_box.png", plt2, path = "../figures")