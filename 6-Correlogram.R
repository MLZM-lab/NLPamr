library("corrplot")
library("gdata")
#install.packages("functional")
library("functional")
#install.packages("writexl")
library("writexl")


####################################
##################Defined functions
####################################

readExcelTable <- function(FILE){
  M_source <- read.xls(FILE, header=FALSE)
  row_names <- M_source[1]
  countryNames <- row_names[-1,1]
  M_source <- M_source[-1]
  feature_names <- M_source[1,]
  M_source <- M_source[-1,]
  row.names(M_source) <- countryNames
  #To change all characters to numeric
  indx <- sapply(M_source, is.character)
  M_source[indx] <- lapply(M_source[indx], function(x) as.numeric(as.character(x)))
  return(list(M_source, feature_names))
}


getPlots <- function(M, p_mat, feature_names, TITLE){ 
  colnames(M) <- feature_names
  row.names(M) <- feature_names
  pdf(TITLE)
  #Now let's order it by the clusters it makes
  corrplot(M, method="color", type="lower", order="hclust", tl.cex=0.4)
  corrplot(M, method="color", order="hclust", tl.cex=0.4)
  #Add significance level to the correlogram. Leave blank on no significant coefficient
  corrplot(M, method="color", tl.cex=0.4, type="lower", order="hclust", p.mat = p_mat, sig.level = 0.05, insig = "blank") #tl.cex=1 <-- for tick label size
  dev.off()
}


getRelevantCorrs <- function(M_papers, M, p_mat, threshold){
  indices <- data.frame(ind= which( (M_papers >= threshold | M_papers <= threshold*-1  ), arr.ind=TRUE))
  if(dim(indices)[1]>0){
    indices$rnm <- t(feature_names[indices$ind.row])
    indices$cnm <- t(feature_names[indices$ind.col])
    corr_val <- c()
    for(i in 1:dim(indices)[1]){
      corr_val <- c(corr_val, M[indices$ind.row[i], indices$ind.col[i]])
    }
    indices<- cbind(indices, corr_val)
    p_val <- c()
    for(i in 1:dim(indices)[1]){
      p_val <-  c(p_val, p_mat[indices$ind.row[i], indices$ind.col[i]])
    }
    indices<- cbind(indices, p_val)    
  }
  return(indices)
}


cor_mtest <- function(mat) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p_mat<- matrix(NA, n, n)
  diag(p_mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j])
      p_mat[i, j] <- p_mat[j, i] <- tmp$p.value
    }
  }
  colnames(p_mat) <- rownames(p_mat) <- colnames(mat)
  p_mat
}

cor_mtest_spearman <- function(mat) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p_mat<- matrix(NA, n, n)
  diag(p_mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], method="spearman")
      p_mat[i, j] <- p_mat[j, i] <- tmp$p.value
    }
  }
  colnames(p_mat) <- rownames(p_mat) <- colnames(mat)
  p_mat
}



computeMatrices<- function(M_source){
  M<-cor(M_source)
  M[!is.finite(M)] <- 0
  #Also compute the p values. Matrix of the p-value of the correlation:
  p_mat <- cor_mtest(M_source)
  p_mat[!is.finite(p_mat)] <- 0
  return(list(M, p_mat))
}



computeMatrices_Spearman<- function(M_source){
  M<-cor(M_source, method="spearman")
  M[!is.finite(M)] <- 0
  #Also compute the p values. Matrix of the p-value of the correlation:
  p_mat <- cor_mtest_spearman(M_source)
  p_mat[!is.finite(p_mat)] <- 0
  return(list(M, p_mat))
}




##########################################################################
##################WHO and WBE + vet + incomeClass dataset
##########################################################################


#Read and process input file
res1 <- readExcelTable("WHO_WBE_Vet_curated_Norm_categories.xlsx")
M_source <- res1[[1]]
feature_names <- res1[[2]]
#Get the correlation and significance matrices
res2 <- computeMatrices(M_source)
M <- res2[[1]]
p_mat <- res2[[2]]
p_mat[] <- p.adjust(p_mat, "fdr")
#Get the plots
getPlots(M,  p_mat, feature_names, "Correlogram_WHO_WBE_Vet_curated_Norm_categories.pdf")
#Get the relevant results for TF-IDF
M_papers <- M[,  c("V2", "V3")] 
indices<- getRelevantCorrs(M_papers, M, p_mat, 0.3)
write_xlsx(indices, "RelevantCorrs_WHO_WBE_Vet_curated_Norm_categories.xlsx")

####With Spearman method instead of the defalt Pearson

res2 <- computeMatrices_Spearman(M_source)
M <- res2[[1]]
p_mat <- res2[[2]]
p_mat[] <- p.adjust(p_mat, "fdr")
#Get the plots
getPlots(M,  p_mat, feature_names, "Correlogram_WHO_WBE_Vet_curated_Norm_categories_Spearman_Spearman.pdf")
#Get the relevant results
M_papers <- M[,  c("V2", "V3")] 
indices<- getRelevantCorrs(M_papers, M, p_mat, 0.3)
write_xlsx(indices, "RelevantCorrs_WHO_WBE_Vet_curated_Norm_categories_Spearman.xlsx")



#################################################################################################
##################Now check it only with all the curated features
#################################################################################################

#Read and process input file
res1 <- readExcelTable("WHO_WBE_Vet_OWiD_curated_Norm_categories.xlsx")
M_source <- res1[[1]]
feature_names <- res1[[2]]

res2 <- computeMatrices(M_source)
M <- res2[[1]]
p_mat <- res2[[2]]
p_mat[] <- p.adjust(p_mat, "fdr")

getPlots(M,  p_mat, feature_names, "Correlogram_WHO_WBE_Vet_OWiD_curated_Norm_categories.pdf")

M_papers <- M[,  c("V2", "V3")] 
indices <- getRelevantCorrs(M_papers, M, p_mat, 0.3)
write_xlsx(indices,"RelevantCorrs_WHO_WBE_Vet_OWiD_curated_Norm_categories.xlsx")

####With Spearman method instead of the defalt Pearson

res2 <- computeMatrices_Spearman(M_source)
M <- res2[[1]]
p_mat <- res2[[2]]
p_mat[] <- p.adjust(p_mat, "fdr")

getPlots(M,  p_mat, feature_names, "Correlogram_WHO_WBE_Vet_OWiD_curated_Norm_categories_Spearman.pdf")

M_papers <- M[,  c("V2", "V3")] 
indices <- getRelevantCorrs(M_papers, M, p_mat, 0.3)
write_xlsx(indices,"RelevantCorrs_WHO_WBE_Vet_OWiD_curated_Norm_categories_Spearman.xlsx")


