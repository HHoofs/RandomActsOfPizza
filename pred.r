require(rms)
require(topicmodels)
require(stringr)
require(plyr)
require(dplyr)
require(stringr)
require(car)
require(rjson)
require(lubridate)
require(slam)
require(tm)
require(SnowballC)
library(XML)


process_cell <- function(cell) {
  if (is.null(cell) || length(cell) == 0) {
    return(NA)
  }
  if (length(cell)>1) {
    return(paste(cell, collapse='; '))    
  }
  return(cell)
}

process_row <- function(row) unlist(lapply(row, process_cell))

getSeason <- function(DATES) {
  WS <- as.Date("2012-12-15", format = "%Y-%m-%d") # Winter Solstice
  SE <- as.Date("2012-3-15",  format = "%Y-%m-%d") # Spring Equinox
  SS <- as.Date("2012-6-15",  format = "%Y-%m-%d") # Summer Solstice
  FE <- as.Date("2012-9-15",  format = "%Y-%m-%d") # Fall Equinox
  
  # Convert dates from any year to 2012 dates
  d <- as.Date(strftime(DATES, format="2012-%m-%d"))
  
  ifelse (d >= WS | d < SE, "Winter",
          ifelse (d >= SE & d < SS, "Spring",
                  ifelse (d >= SS & d < FE, "Summer", "Fall")))
}

remove_HTML_markup <- function(s) tryCatch({
    doc <- htmlTreeParse(paste("<!DOCTYPE html>", s),
                         asText = TRUE, trim = FALSE)
    xmlValue(xmlRoot(doc))
  }, error = function(s) s)

#### Data ################
# To get started we're load the training data and convert it from JSON to a DataFrame
json <- fromJSON(file='train.json')

# final data
train <- as.data.frame(do.call("rbind", 
                               lapply(json, process_row)))

# factor to character
train <-
  train %>% 
  mutate_if(is.factor,as.character)

# head
head(train)

# transform to numeric
cont_var <- c("number_of_downvotes_of_request_at_retrieval","number_of_upvotes_of_request_at_retrieval","request_number_of_comments_at_retrieval",
              "requester_account_age_in_days_at_request","requester_account_age_in_days_at_retrieval","requester_days_since_first_post_on_raop_at_request",
              "requester_days_since_first_post_on_raop_at_retrieval","requester_number_of_comments_at_request","requester_number_of_comments_at_retrieval",
              "requester_number_of_comments_in_raop_at_request","requester_number_of_comments_in_raop_at_retrieval","requester_number_of_posts_at_request",
              "requester_number_of_posts_on_raop_at_retrieval","requester_number_of_subreddits_at_request","requester_upvotes_minus_downvotes_at_request",
              "requester_upvotes_minus_downvotes_at_retrieval","requester_upvotes_plus_downvotes_at_request","requester_upvotes_plus_downvotes_at_retrieval",
              "requester_number_of_posts_on_raop_at_request")

for(i in cont_var){
    train[,i] <- as.numeric(train[,i])
}

# transform logical to numeric
log_var <- c("requester_received_pizza")

require(car)

for(i in log_var){
  train[,i] <- recode(train[,i],"'FALSE'=0;'TRUE'=1") 
}

# ID
train$ID <- as.numeric(rownames(train))

# Date
train$date <- as.POSIXct(as.numeric(train$unix_timestamp_of_request_utc), origin="1970-01-01",tz = "US/Central")

# time
train$time <- hour(train$date) + minute(train$date) / 60

# weekday
train$day  <- wday(train$date,label=FALSE)

# season
train$season <- getSeason(train$date)

# nchar title/texty
train$title_nchar <- nchar(train$request_title)
train$textr_nchar <- nchar(train$request_text_edit_aware)

# former request dich
train$form_request <- Recode(train$requester_number_of_posts_on_raop_at_request,"0=0;1:10=1")

# nsfw sub
nsfw <- readChar("nsfw_r.txt", file.info("nsfw_r.txt")$size)

nsfw <- str_split(nsfw," ")[[1]]
nsfw <- nsfw[nchar(nsfw) > 0]

train$nsfw_sub <- 0

for(i in 1:nrow(train)){
  all_sub <- train[i,"requester_subreddits_at_request"]
  if(!is.na(all_sub)){
    all_sub <- str_split(all_sub,"; ")[[1]]
    if(any(sapply(all_sub, function(x) any(x==nsfw)))) train[i,"nsfw_sub"] <- 1
  }
}

# linking
train$imgemb_ev <- as.numeric(str_detect(train$request_text_edit_aware,"imgur")|str_detect(train$request_text_edit_aware,"\\[")|str_detect(train$request_text_edit_aware,"http"))

#### Topic models ######################
text_awareD <- data.frame(ID=train$ID,text_aware=as.character(train$request_text_edit_aware),stringsAsFactors = FALSE)

text_awareD <- text_awareD[sapply(text_awareD[, "text_aware"],Encoding) == "unknown",]

ID_sel <- text_awareD$ID

text_corpus <- Corpus(VectorSource(sapply(text_awareD$text_aware,remove_HTML_markup)))

Sys.setlocale("LC_COLLATE", "C")

text_aware_dtm <- DocumentTermMatrix(text_corpus,
                              control = list(stemming = TRUE, stopwords = TRUE, minWordLength = 3,
                                             removeNumbers = TRUE, removePunctuation = TRUE))

dim(text_aware_dtm)

summary(col_sums(text_aware_dtm))

term_tfidf <-
  tapply(text_aware_dtm$v/row_sums(text_aware_dtm)[text_aware_dtm$i], text_aware_dtm$j, mean) *
  log2(nDocs(text_aware_dtm)/col_sums(text_aware_dtm > 0))
summary(term_tfidf)

text_aware_dtm <- text_aware_dtm[,term_tfidf >= 0.1]
text_aware_dtm <- text_aware_dtm[row_sums(text_aware_dtm) > 0,]
summary(col_sums(text_aware_dtm))

dim(text_aware_dtm)

k <- 5
text_aware_TM <- LDA(text_aware_dtm, k = k, method = "Gibbs", control = list(seed = 2016, burnin = 1000, thin = 100, iter = 1000))

terms(text_aware_TM,20)

text_aware_DF <- data.frame(topics(text_aware_TM))
text_aware_DF$ID <- ID_sel[as.numeric(rownames(text_aware_DF))]
names(text_aware_DF) <- c("top","ID")

train <- merge(train,text_aware_DF,by="ID",all.x=TRUE)

train$top[is.na(train$top)] <- sample(1:5,sum(is.na(train$top)),replace = TRUE)

#### Univariate #####################
s <- summary(requester_received_pizza ~ top + imgemb_ev + nsfw_sub + season + day + form_request, data = train )
plot (s , main = "Train (RaoP)", subtitles = FALSE ) 

#### Pred model #####################
train_pred <- train

to_factor <- c("day","top")

# Transfor variables that are not continious to a factor (in order to use them correctly in the regression analysis)
for(i in to_factor){
  # print(i)
  # select if not continu defined in the VarBook
    # Recode from 1 to the number of values and make into factor
    train_pred[,i] <- factor(as.numeric(factor(train_pred[,i] + (1 - min(train_pred[,i],na.rm = TRUE)))))
    # Check is factor is defined correctly (e.g. Factor w/ 4 levels "1","2","3","4": 1 3 2 2 1 3 1 1 1 1 ...) in which the first number should be "1"
    if(attributes(train_pred[,i])$levels[1] != 1) stop("Factor labeling gone wrong")
}

form <- "requester_received_pizza ~ top + imgemb_ev + season + rcs(time,4) + day + form_request + rcs(textr_nchar,3) + nsfw_sub + 
                                       rcs(requester_number_of_comments_at_request,3) +
                                       rcs(requester_account_age_in_days_at_request,3) +
                                       rcs(requester_days_since_first_post_on_raop_at_request,3) +
                                       rcs(requester_upvotes_plus_downvotes_at_retrieval,3)+
                                       rcs(requester_number_of_subreddits_at_request,3)"

ddist <- datadist(train_pred)
options(datadist='ddist')

Fit.Reg  <- lrm(formula = as.formula(form),data = train_pred, x = TRUE, y = TRUE)
## Validate using bw (bootstrap validation of regression model incl predictor selection strategy) 
Fit.Val <- validate(Fit.Reg, B = 1000, type="individual", bw = TRUE, rule="p", sls=.5, estimates = TRUE, pr = TRUE)

Fit.SelIV  <- attributes(Fit.Val)$kept
Fit.SelIVc <- colSums(Fit.SelIV)
Fit.SelIVc <- Fit.SelIVc/1000

Data.Ret <- data.frame(Var=names(Fit.SelIVc),Ret=Fit.SelIVc)
Data.Ret$Var <- factor(Data.Ret$Var,levels = Data.Ret$Var[order(-Data.Ret$Ret)],ordered = TRUE)
# Data.Ret$Keep <- as.numeric(Data.Ret$Ret > Options.Per * (Options.N_boot * Options.N_imp))
# Data.Ret$Keep[1:length(Var.Forced)] <- 2
Data.Ret <- Data.Ret[!is.na(Data.Ret$Var),]

# pdf(paste(Text.Prefix,"//RetentionPlot.pdf",sep=""),width =  11.69, height = 11.69)
ggplot(Data.Ret,aes(x=factor(Var),y=Ret)) +
  geom_bar(stat="identity",colour="black") +
  scale_x_discrete("") + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position="none")
# dev.off()

formF <- "requester_received_pizza ~ top + imgemb_ev + season + rcs(time,4) + day + form_request + rcs(textr_nchar,3) + 
                                       rcs(requester_days_since_first_post_on_raop_at_request,3) +
                                       rcs(requester_number_of_subreddits_at_request,3) + 
                                       rcs(requester_days_since_first_post_on_raop_at_request,3)"

Fit.RegF  <- lrm(formula = as.formula(formF),data = train_pred, x = TRUE, y = TRUE)
 
Final.ValF <- validate(Fit.RegF,B = 1000)
Final.ValF <- rbind(Final.ValF,(Final.ValF[1,]+1)/2)
rownames(Final.ValF)[nrow(Final.ValF)] <- "C"
Final.ValF["C","optimism"] <- Final.ValF["C","optimism"] - .5
Final.ValF["C","n"] <- 1000

write.table(round(Final.ValF,3),"Validate.csv",sep = ";",row.names = TRUE,col.names = NA,dec = ".")

#### EXTERNAL: Data ########################
# To get started we're load the training data and convert it from JSON to a DataFrame
json <- fromJSON(file='test.json')

# final data
test <- as.data.frame(do.call("rbind", lapply(json, process_row)))

# factor to character
test <-
  test %>% 
  mutate_if(is.factor,as.character)

# head
head(test)

# transform to numeric
cont_var <- c("number_of_downvotes_of_request_at_retrieval","number_of_upvotes_of_request_at_retrieval","request_number_of_comments_at_retrieval",
              "requester_account_age_in_days_at_request","requester_account_age_in_days_at_retrieval","requester_days_since_first_post_on_raop_at_request",
              "requester_days_since_first_post_on_raop_at_retrieval","requester_number_of_comments_at_request","requester_number_of_comments_at_retrieval",
              "requester_number_of_comments_in_raop_at_request","requester_number_of_comments_in_raop_at_retrieval","requester_number_of_posts_at_request",
              "requester_number_of_posts_on_raop_at_retrieval","requester_number_of_subreddits_at_request","requester_upvotes_minus_downvotes_at_request",
              "requester_upvotes_minus_downvotes_at_retrieval","requester_upvotes_plus_downvotes_at_request","requester_upvotes_plus_downvotes_at_retrieval",
              "requester_number_of_posts_on_raop_at_request")

for(i in cont_var){
  if(any(names(test) == i)){
    print(i)
    test[,i] <- as.numeric(test[,i])
  }
}

# transform logical to numeric
log_var <- c("requester_received_pizza")

require(car)

for(i in log_var){
  test[,i] <- recode(test[,i],"'FALSE'=0;'TRUE'=1") 
}

# ID
test$ID <- as.numeric(rownames(test))

# Date
test$date <- as.POSIXct(as.numeric(test$unix_timestamp_of_request_utc), origin="1970-01-01",tz = "US/Central")

# time
test$time <- hour(test$date) + minute(test$date) / 60

# weekday
test$day  <- wday(test$date,label=FALSE)

# season
test$season <- getSeason(test$date)

# nchar title/texty
test$title_nchar <- nchar(test$request_title)
test$textr_nchar <- nchar(test$request_text_edit_aware)

# former request dich
test$form_request <- Recode(test$requester_number_of_posts_on_raop_at_request,"0=0;1:10=1")

# nsfw sub
nsfw <- readChar("nsfw_r.txt", file.info("nsfw_r.txt")$size)

nsfw <- str_split(nsfw," ")[[1]]
nsfw <- nsfw[nchar(nsfw) > 0]

test$nsfw_sub <- 0

for(i in 1:nrow(test)){
  all_sub <- test[i,"requester_subreddits_at_request"]
  if(!is.na(all_sub)){
    all_sub <- str_split(all_sub,"; ")[[1]]
    if(any(sapply(all_sub, function(x) any(x==nsfw)))) test[i,"nsfw_sub"] <- 1
  }
}

# linking
test$imgemb_ev <- as.numeric(str_detect(test$request_text_edit_aware,"imgur")|str_detect(test$request_text_edit_aware,"\\[")|str_detect(test$request_text_edit_aware,"http"))

#### EXTERNAL: Topic models ######################
text_awareD <- data.frame(ID=test$ID,text_aware=as.character(test$request_text_edit_aware),stringsAsFactors = FALSE)
# text_awareD <- text_awareD[sapply(text_awareD$text_aware,Encoding) == "unknown"]

text_awareD <- text_awareD[sapply(text_awareD[, "text_aware"],Encoding) == "unknown",]

ID_sel <- text_awareD$ID

text_corpus <- Corpus(VectorSource(sapply(text_awareD$text_aware,remove_HTML_markup)))

Sys.setlocale("LC_COLLATE", "C")

text_aware_dtm <- DocumentTermMatrix(text_corpus,
                              control = list(stemming = TRUE, stopwords = TRUE, minWordLength = 3,
                                             removeNumbers = TRUE, removePunctuation = TRUE))

dim(text_aware_dtm)

summary(col_sums(text_aware_dtm))

term_tfidf <-
  tapply(text_aware_dtm$v/row_sums(text_aware_dtm)[text_aware_dtm$i], text_aware_dtm$j, mean) *
  log2(nDocs(text_aware_dtm)/col_sums(text_aware_dtm > 0))
summary(term_tfidf)

text_aware_dtm <- text_aware_dtm[,term_tfidf >= 0.1]
text_aware_dtm <- text_aware_dtm[row_sums(text_aware_dtm) > 0,]
summary(col_sums(text_aware_dtm))

dim(text_aware_dtm)

prop_top <- posterior(text_aware_TM,text_aware_dtm)

test$top <- 99

for(i in 1:nrow(prop_top$topics)){
  print(i)
  IDfi <- rownames(prop_top$topics)[i]
  propt_Y <- prop_top$topics[i,]
  top_sel <- which(max(propt_Y) == propt_Y)
  if(length(top_sel) > 1) top_sel <- top_sel[sample(1:length(top_sel),1)]
  test[test$ID == ID_sel[as.numeric(IDfi)],"top"] <- top_sel
}

test$top[test$top==99] <- sample(1:5,sum(test$top==99),replace = TRUE)

val.prob()

coef_new <- coef(Fit.RegF)[-1]*Final.ValF[4,"index.corrected"]
lp_new <- Fit.RegF$x %*% coef_new
intercept_new <- lrm(train$requester_received_pizza ~ offset(lp_new))

it.RegF <- Fit.RegF

it.RegF$coefficients <- c(coef(intercept_new),coef_new)

pred.logit2 <- predict(Fit.RegF, test)
phat <- 1/(1+exp(-pred.logit2))

require(pROC)

plot.roc(train[,"requester_received_pizza"],phat,smooth=TRUE,ci=TRUE)


pred.logit2 <- predict(Fit.RegF, train)
phat2 <- 1/(1+exp(-pred.logit2))


rocobj <- plot.roc(train[,"requester_received_pizza"],
                   phat2,  
                   main="Confidence intervals", 
                   percent=TRUE,  
                   smooth=FALSE,
                   ci=FALSE, # compute AUC (of AUC by default)
                   print.auc=TRUE) 

spec_sens <- data.frame(spec=rocobj$specificities,sens=rocobj$sensitivities)

ggplot(spec_sens,aes(x=100-spec,y=sens))+geom_line(size = 2, alpha = 1)+
  labs(title= "ROC curve", 
       x = "Specificity", 
       y = "Sensitivity") +
  scale_x_continuous(breaks=c(0,25,50,75,100),labels=c(100,75,50,25,0)) +
  geom_abline(intercept = 0, slope = 1) +
  theme_bw()

ciobj <- ci.se(rocobj, # CI of sensitivity  
               specificities=seq(0, 100, 5)) # over a select set of specificities  

plot(ciobj, type="shape", col="#1c61b6AA") # plot as a blue shape  
# plot(ci(rocobj, of="thresholds", thresholds="best")) # add one threshold

wordcloud()

pred.logit <- predict(it.RegF, test)
phat <- 1/(1+exp(-pred.logit))

test_out <- data.frame(request_id=test$request_id,requester_received_pizza=NA)

test_out$requester_received_pizza <- phat

write.csv(test_out,row.names = FALSE,file="submission_hhoofs.csv")
