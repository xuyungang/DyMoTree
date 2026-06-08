if (!requireNamespace("BiocManager", quietly=TRUE)) {
  install.packages("BiocManager")
}
BiocManager::install("splatter")
BiocManager::install("scater")

library(splatter)
library(scater)

params <- newSplatParams()
params <- setParam(params, "nGenes", 1000)
params <- setParam(params, "batchCells", 5000)
sim <- splatSimulatePaths(
  seed=801,
  params,
  group.prob = c(2,2,2,3,3)/sum(c(2,2,2,3,3)),  
  de.prob =0.5,  
  path.from = c(0,1,1,2,3),  
  path.nSteps = 10000,
  de.facScale = c(0.2, 0.4, 0.4, 0.2, 0.2),
  path.skew=1,  
  bcv.common=1,
  out.prob=0.001,
  verbose = FALSE
)
sim <- logNormCounts(sim)
sim <- runPCA(sim)
plotPCA(sim, colour_by = "Group") + ggtitle("Branching path")




in.silico.count <- sim@assays@data@listData[["counts"]]
in.silico.meta <- as.data.frame(sim@colData)
in.silico.seurat <- sc_processing(in.silico.count,in.silico.meta)

in.silico.seurat@meta.data$celltype <- ifelse(in.silico.meta$Group%in%paste('Path',1:3,sep = ''),'Stem',
                                                                          ifelse(in.silico.meta$Group=='Path4','Child1','Child2'))
in.silico.seurat@meta.data$label <- ifelse(in.silico.seurat$Group%in%c('Path4','Path5'),'Descendant',ifelse(in.silico.seurat$Group=='Path2','Fate1',ifelse(in.silico.seurat$Group=='Path3','Fate2','Stemness')))
in.silico.seurat$Dim_1 <- in.silico.seurat@reductions$umap@cell.embeddings[,1]
in.silico.seurat$Dim_2 <- in.silico.seurat@reductions$umap@cell.embeddings[,2]

rm(in.silico.meta)
rm(in.silico.count)
rm(sim)
rm(params)

DimPlot(in.silico.seurat,group.by = 'label')+DimPlot(in.silico.seurat,group.by = 'celltype')


is.list <- SplitObject(in.silico.seurat,split.by = 'celltype')

task <- 'tree1'
if (!dir.exists(paste('./scLT/Simulated/',task,sep = ''))){
  dir.create(paste('./scLT/Simulated/',task,sep = ''))
}
setwd(paste('./scLT/Simulated/',task,sep = ''))

for(i in 1:length(names(is.list))){
  cell <- names(is.list)[i]
  obs <- is.list[[cell]]@meta.data
  pca <- is.list[[cell]]@reductions$pca@cell.embeddings[,1:20]
  mat <- t(as.matrix(is.list[[cell]]@assays$RNA$data[VariableFeatures(is.list[[cell]]),]))
  write.csv(obs,paste(task,cell,'obs.csv',sep = '.'))
  write.csv(pca,paste(task,cell,'emb.csv',sep = '.'))
  write.csv(mat,paste(task,cell,'mat.csv',sep = '.'))
}

rm(i)
rm(cell)
rm(task)

saveRDS(in.silico.seurat,file = 'srt.obj.rdata')

