library(survival)
library(survminer)
library(dplyr)
library(tidyverse)

library(clusterProfiler)
library(org.Hs.eg.db)

data <- readRDS('./pocessed_TCGAPanCan_LUAD.rdata')
gene <- read.csv('../CPTAC/LUAD.driver.csv')

at1 <- gene %>% arrange(desc(AT1_fate)) %>% dplyr::select(X)
emt <- gene %>% arrange(desc(EMT_fate)) %>% dplyr::select(X)

gene_set <- toupper(at1$X[1:30])


entrez_conversion <- bitr(toupper(gene_set), 
                          fromType = "SYMBOL", 
                          toType = c("ENTREZID"),
                          OrgDb = org.Hs.eg.db)
gene_set <- entrez_conversion$ENTREZID

exp_matrix <- data[,gene_set]
exp_matrix$patient_id <- rownames(exp_matrix)


final_data <- data[,c(colnames(data)[1:4],gene_set)]
final_data$gene_set_score <- rowMeans(final_data[, -c(1:4), drop=FALSE], na.rm = TRUE)
  

median_score <- median(final_data$gene_set_score, na.rm = TRUE)
final_data$score_group <- ifelse(final_data$gene_set_score >= median_score, "high", "low")
final_data$score_group <- factor(final_data$score_group, levels = c("low", "high"))
  
surv_object_set <- Surv(time = final_data$`Overall Survival (Months)`, event = final_data$Status)
fit_set <- survfit(surv_object_set ~ score_group, data = final_data)
ggsurvplot(
  fit_set,
  data = final_data,
  pval = TRUE,                                    
  risk.table = TRUE,              
  legend.labs = c("low", "high"),
  legend.title = "AT1_fate",
      xlab = "time(month)",
  ylab = "Percent survival",
  palette = c("#066190", "#C42238"), 
  surv.median.line = 'hv',
  conf.int.style='step',
  fontsize=5,
  font.legend=15,
  font.x='bold',
  font.y='bold'
      
)
