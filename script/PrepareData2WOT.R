if(FALSE){"
  处理时间序列单细胞数据，用于Waddington-OT算法输入数据
  细胞注释需要包含 id days celltype x y"
}

# 读取数据
#srt <- readRDS('D:/scLT/Larry/srt.with.clone.rdata')
setwd('./scLT/Weinreb/WOT_data/day246')
#sub.srt <- readRDS("./srt.rdata")
#1. 制作count file
#write.table(t(as.matrix(sub.srt@assays$RNA$counts)), file = "./Data.WOT/counts.mtx",quote = F,row.names = T,col.names = T,sep = '\t')

#2. 制作所有所需的注释文件
sub.srt <- readRDS('./scLT/Weinreb/0814/larry.desc246.rdata')
sub.srt <- merge(sub.srt$HSPC,sub.srt[-1])
sub.srt <- JoinLayers(sub.srt)
sub.srt$Time <- ifelse(sub.srt$Cell.type.annotation=='Undifferentiated',1,2)
sub.srt <- NormalizeData(sub.srt,normalization.method = 'LogNormalize')
sub.srt <- FindVariableFeatures(sub.srt)

cell_meta <- sub.srt@meta.data %>% dplyr::select(Time,Cell.type.annotation,SPRING.x,SPRING.y)
cell_meta$id <- paste('X',rownames(cell_meta),sep = '')
colnames(cell_meta) <- c('day','celltype','x','y','id')


#3.制作days.txt
write.table(cell_meta %>% dplyr::select(id,day),sep = '\t',quote = F,row.names = F,col.names = T,file = './days.txt')

#4.制作cell id
write.table(cell_meta %>% dplyr::select(id),quote = F,row.names = F,col.names = F,sep = '\t',file = './serum_cell_ids.txt')

#5. 制作细胞类型下标文件
library(dplyr)

gmx_list <- cell_meta %>%
  group_by(celltype) %>%
  summarise(cell_ids = list(id), .groups = "drop") %>%
  mutate(dash = "-")  # 添加第二列 "-"

# 转换为矩阵格式，每一行代表一个 celltype
gmx_data <- do.call(rbind, lapply(1:nrow(gmx_list), function(i) {
  c(gmx_list$celltype[i], gmx_list$dash[i], unlist(gmx_list$cell_ids[i]))
}))

# 写出 GMX 文件（制表符分隔）
write.table(gmx_data, file = "./major_cell_sets.gmt", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)


#6. 制作HVG数据
mat <- t(as.matrix(sub.srt@assays$RNA$data[VariableFeatures(sub.srt),]))
rownames(mat) <- cell_meta$id
colnames(mat) <- VariableFeatures(sub.srt)
write.table(mat, file = "./hvg.counts.txt",quote = F,sep = '\t',row.names = T,col.names = T)




# 7. growth rate
cell_meta$cell_growth_rate <- 1
write.table(cell_meta %>% dplyr::select(id,cell_growth_rate),quote = F,row.names = F,col.names = T,sep = '\t',file = './growth_gs_init.txt')
cell_meta1 <- sub.srt@meta.data
rownames(cell_meta1) <- paste('X',rownames(cell_meta1),sep = '')
write.csv(cell_meta1,file = './meta.data.csv')
rm(cell_meta)
rm(cell_meta1)
rm(mat)
rm(gmx_list)
rm(gmx_data)





library(dplyr)
setwd('./scLT/Weinreb/WOT_data/day246/res/')
sub.srt <- readRDS('./scLT/Weinreb/0814/larry.desc246.rdata')
sub.srt <- merge(sub.srt$HSPC,sub.srt[-1])
sub.srt <- JoinLayers(sub.srt)
colnames(sub.srt) <- paste('X',colnames(sub.srt),sep = '')

mo <- read.table('./Monocyte_fate_fates.txt',header = T,row.names = 1)
neu <- read.table('./Neutrophil_fate_fates.txt',header = T,row.names = 1)
wot_res <- cbind(mo %>% dplyr::select(Monocyte),neu %>% dplyr::select(Neutrophil))

wot_res <- wot_res[colnames(sub.srt),]
sub.srt$Monocyte_fate <- wot_res$Monocyte
sub.srt$Neutrophil_fate <- wot_res$Neutrophil
#sub.srt <- NormalizeData(sub.srt,normalization.method = 'LogNormalize')
#sub.srt <- FindVariableFeatures(sub.srt)
#sub.srt <- ScaleData(sub.srt)
#sub.srt <- RunPCA(sub.srt)
#sub.srt <- RunUMAP(sub.srt,dims=1:30)
#sub.srt@reductions$umap@cell.embeddings[,1] <- sub.srt$SPRING.x
#sub.srt@reductions$umap@cell.embeddings[,2] <- sub.srt$SPRING.y

res <- sub.srt@meta.data %>% dplyr::select(Cell.type.annotation,Weinreb_fate,Monocyte_fate,Neutrophil_fate)
rename_X_to_cell <- function(df, prefix = "cell") {
  old_names <- rownames(df)
  
  # 只替换以 X + 数字 开头的列名（避免误伤）
  new_names <- ifelse(
    grepl("^X[0-9]+$", old_names),
    sub("^X", prefix, old_names),
    old_names
  )
  
  rownames(df) <- new_names
  return(df)
}

res <- rename_X_to_cell(res)
res <- res %>% filter(Cell.type.annotation=='Undifferentiated')

write.csv(res,file = 'wot.res.csv',quote = F,col.names = T,row.names = T)
