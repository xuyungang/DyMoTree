
sub.srt <- sub.srt[,sub.srt@meta.data %>% filter(celltype%in%c('HSPC','Monocyte','Neutrophil')) %>% rownames()]
sub.srt <- NormalizeData(sub.srt,normalization.method = 'LogNormalize')
sub.srt <- FindVariableFeatures(sub.srt)
sub.srt <- ScaleData(sub.srt)
sub.srt <- RunPCA(sub.srt)
srt.list <- SplitObject(sub.srt,split.by = 'cell')
bitree <- c('HSPC','Monocyte','Neutrophil','Baso_Mast','Ery_Meg')
setwd('./scLT/Weinreb/PolyTree/Data.DyMoTree/')
for(i in 1:length(bitree)){
  obs <- srt.list[[bitree[i]]]@meta.data
  pca <- srt.list[[bitree[i]]]@reductions$pca@cell.embeddings[,1:20]
  mat <- t(as.matrix(srt.list[[bitree[i]]]@assays$RNA$data[VariableFeatures(srt.list[[bitree[i]]]),]))
  write.csv(obs,paste('Larry.',bitree[i],'.obs.csv',sep = ''))
  write.csv(pca,paste('Larry.',bitree[i],'.emb.csv',sep = ''))
  write.csv(mat,paste('Larry.',bitree[i],'.mat.csv',sep = ''))
}
rm(i)
rm(obs)
rm(pca)
rm(mat)
rm(bitree)