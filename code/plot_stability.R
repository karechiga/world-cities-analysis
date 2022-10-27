# This script plots the stability of the cities in the given dataset

library(ggplot2)

# read in the dataset
data <- read.csv("../cluster_data/cities.csv")
data$Cluster <- factor(data$Cluster)

# Creates a boxplot
plt1 <- ggplot(data, aes(x=Cluster,group=Cluster,y = Stability*100, color=Cluster,shape=Cluster)) + 
  geom_boxplot(outlier.shape = NA) + geom_jitter(alpha=0.17, width=0.3) +
  xlab("Cluster") + ylab("Stability (%)") +
  labs(title=paste("Stability of K-Means Clustered Cities Boxplots (k =", length(unique(data$Cluster)), ")"),) +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 12), plot.title = element_text(size = 15) )

show(plt1)

# saving the plots as pngs in the "figures" folder 
ggsave(paste("Cluster_Stability_", length(unique(data$Cluster)) ,".png", sep=""), plt1, path = "../figures")
