install.packages("devtools")
library(devtools)
install_github("dgrun/FateID")
library(FateID)
vignette("FateID")
library(Seurat)
library(dplyr)

#load data
#data(intestine)
setwd('D:/scLT/Weinreb/0814/desc6.1/')
sub.srt <- readRDS('D:/scLT/Weinreb/0814/larry.desc6.rdata')
sub.srt <- merge(sub.srt$HSPC,sub.srt[-1])
sub.srt <- JoinLayers(sub.srt)
sub.srt <- NormalizeData(sub.srt,normalization.method = 'LogNormalize')
sub.srt <- FindVariableFeatures(sub.srt)
sub.srt <- ScaleData(sub.srt)
sub.srt <- RunPCA(sub.srt)


x <- as.data.frame(sub.srt@assays$RNA$data[VariableFeatures(sub.srt),])
#intestine$x# gene express data must be a data.frame with gene for row, cell for colum
y <- sub.srt@meta.data$Cell.type.annotation
y[which(y=='Undifferentiated')] <- 0
y[which(y=='Monocyte')] <- 1
y[which(y=='Neutrophil')] <- 2


y <- as.numeric(y)
names(y) <- colnames(x)
#intestine$y# The label has to be provided as a vector with integer values and component names corresponding to column names of the expression data frame
tar <- c(1:2)
fb  <- fateBias(x, y, tar, seed=12345,minnr = 5,minnrh = 20)


m3 <- sub.srt@meta.data
m3$Monocyte_fate <- fb$probs$t1
m3$Neutrophil_fate <- fb$probs$t2
m3 <- m3 %>% filter(Cell.type.annotation=='Undifferentiated')
m3$fate_bias <- (m3$Monocyte_fate)/(m3$Monocyte_fate+m3$Neutrophil_fate)

m3 <- m3 %>% dplyr::select(Cell.type.annotation,Weinreb_fate,fate_bias)
colnames(m3)[2] <- 'ground_truth'
rownames(m3) <- paste('cell',rownames(m3),sep='')


write.csv(m3,file = './fateid.res.csv')
