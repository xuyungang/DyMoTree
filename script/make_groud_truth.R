# Load data
srt <- readRDS("./scLT/Larry/PolyTree/srt.with.clone.rdata")

# Parameters
source_type <- "HSPC"                 # progenitor / source population
target_types <- c("Monocyte", "Neutrophil")  # target fates

# Extract metadata
meta <- srt@meta.data

# Split populations
source_cells <- meta %>% dplyr::filter(celltype == source_type)
desc_cells   <- meta %>% dplyr::filter(celltype != source_type)

# Initialize result
fate_bias <- numeric(nrow(source_cells))

for (i in seq_len(nrow(source_cells))) {
  bc <- source_cells$barcodes[i]
  idx <- which(desc_cells$barcodes == bc)
  
  if (length(idx) == 0) {
    fate_bias[i] <- 0
    next
  }
  
  stats <- table(desc_cells$celltype[idx])
  
  counts <- sapply(target_types, function(t) {
    if (t %in% names(stats)) stats[[t]] else 0
  })
  
  total <- sum(counts)
  
  fate_bias[i] <- if (total == 0) {
    0
  } else {
    as.numeric(counts[1] / total) + 1e-4
  }
}

# Assign result
source_cells$fate_bias <- fate_bias

meta$Weinreb_fate <- 0
meta[rownames(source_cells), "Weinreb_fate"] <- source_cells$fate_bias

srt@meta.data <- meta

# Cleanup
rm(fate_bias, source_cells, desc_cells, meta)