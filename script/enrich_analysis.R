# 加载必要的包
library(clusterProfiler)
library(org.Hs.eg.db)   # 如果是小鼠数据（Mus musculus）
library(org.Mm.eg.db) # 如果是人类数据，取消这一行注释

library(enrichplot)
library(ggplot2)
library(DOSE)

# 转置矩阵，使样本为行，基因为列

genes <- read.csv('../CPTAC/LUAD.driver.csv')
at1 <- genes %>% arrange(desc(AT1_fate)) %>% dplyr::select(X)
emt <- genes %>% arrange(desc(EMT_fate)) %>% dplyr::select(X)


# 4️⃣ GO 富集分析
at1_ego <- enrichGO(gene         = at1$X[1:30],
                OrgDb        = org.Mm.eg.db,  # 或 org.Hs.eg.db
                keyType      = "SYMBOL",
                ont          = "BP",   # 可改为 "BP" "CC" "MF"
                pAdjustMethod= "BH",
                pvalueCutoff = 0.05,
                qvalueCutoff = 0.05)
emt_ego <- enrichGO(gene         = emt$X[1:30],
                    OrgDb        = org.Mm.eg.db,  # 或 org.Hs.eg.db
                    keyType      = "SYMBOL",
                    ont          = "BP",   # 可改为 "BP" "CC" "MF"
                    pAdjustMethod= "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.05)



at1_ego@result$Fate_bias <- 'AT1-fate'
emt_ego@result$Fate_bias <- 'EMT-fate'

res <- rbind(emt_ego@result,at1_ego@result)
res <- res %>% 
  group_by(Fate_bias) %>% 
  top_n(n = 5, wt = Count) 

res$ID <- factor(res$ID,levels=res$ID)


res %>% 
  ggplot(aes(x = ID,
             y = Count)) + 
  geom_col(aes(fill = Fate_bias),
           width = 0.65,
           position = "dodge"
  ) + 
  scale_y_continuous(expand = c(0,0)) + 
  scale_x_discrete(expand = c(0.04,0)) + 
  theme_classic() + 
  coord_flip()+ 
  geom_text(aes(y = 0.1, label = Description),
            hjust = 0,
            fontface = "italic",
            size = 5) +
  scale_color_manual(
    values = c("#066190","#c42238"))+
  scale_fill_discrete(type = c("#066190","#c42238"))


