library(tidyr)
library(ggplot2)
library(ggforce)
library(ggpubr)
library(readr)
library(dplyr)
library(readr)
library(tidyverse)
library(ggpointdensity)
library(RColorBrewer)
library(scales)

#Import the shap value table and the corresponding normalized eigenvalue table
options(scipen = 200)
data <- read.csv("shap_value.csv")
df <- read.csv("frature_value.csv")
# Wide Data to Long Data
data_long<-gather(data, type, shap_value, `Industrial_sector`:`Zn`)
df_long<-gather(df, type, feature_value, `Industrial_sector`:`Zn`)
#Plus eigenvalues
data_long$feature_value<-df_long$feature_value
#Ranged by the MAS of the features
shap_mean<-as.data.frame(arrange(aggregate(abs(data_long$shap_value), by=list(type=data_long$type),mean), x))
names(shap_mean)[2]<-'value'
write.csv(shap_mean, "data_MAS.csv")




#Specify the x-axis factor level
level<-shap_mean$type
windowsFonts(A=windowsFont("Arial"))

#Take the MAS of each feature as the importance of that feature
p1 = ggplot(shap_mean[13:32,], aes(x=factor(type,levels=level),y = value,fill=type))+ 
  theme_bw(base_family = "A")+
  labs(title="", x="", y="Feature Importance")+
  geom_col(width = 0.5) +
  scale_fill_manual(
  values=c(
  "water"="#A12345","N"="#A12345",
  "staff"="#FFB703","process_9"="#5E4FA2",
  "COD"="#A12345",
  "process_2"="#5E4FA2","NH3N"="#A12345",
  "pH"="#A12345","process_15"="#5E4FA2",
  "process_10"="#5E4FA2","P"="#A12345",
  "process_7"="#5E4FA2","Cu"="#A12345",
  "process_11"="#5E4FA2","process_13"="#5E4FA2",
  "Fe"="#A12345", "process_16"="#5E4FA2",
  "process_5"="#5E4FA2","process_17"="#5E4FA2"))+
  geom_text(aes(label = round(value, 2)), position = position_dodge(0.6), hjust=-0.3,vjust = 0.3,size=4.8) +
  scale_y_continuous(breaks = seq(0,0.16,0.04),expand = c(0.005,0))+
  coord_flip(ylim = c(0,0.16))+
  theme(plot.title=element_text(face="bold",
                                size=18, color="black", hjust = 0.5),
        axis.title = element_text(face="bold",size=15,color="black"),
        axis.text=element_text(size=13,color="black"),
        panel.background=element_rect(fill="white",
                                      color="black"),
        legend.position = ' none ',
        panel.grid=element_blank(),
        plot.margin = unit(c(1.2,1.5,0.8,1.5), 'lines'))

Cairo::CairoPDF("Importance.pdf",
                width = 6,
                height = 9)
p1
dev.off()
