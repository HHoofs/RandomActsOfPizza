# To get started we're load the training data and convert it from JSON to a DataFrame
library(rjson)

json <- fromJSON(file='train.json')

# Helper functions for converting the JSON data to a dataframe
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

train <- as.data.frame(do.call("rbind", lapply(json, process_row)))

library(dplyr)

train <-
  train %>% 
  mutate_if(is.factor,as.character)

# train %>% mutate_if(is.factor, as.character) -> train

head(train)

cont_var <- c("number_of_downvotes_of_request_at_retrieval",
              "number_of_upvotes_of_request_at_retrieval",
              "request_number_of_comments_at_retrieval",
              "requester_account_age_in_days_at_request",
              "requester_account_age_in_days_at_retrieval",
              "requester_days_since_first_post_on_raop_at_request",
              "requester_days_since_first_post_on_raop_at_retrieval",
              "requester_number_of_comments_at_request",
              "requester_number_of_comments_at_retrieval",
              "requester_number_of_comments_in_raop_at_request",
              "requester_number_of_comments_in_raop_at_retrieval",
              "requester_number_of_posts_at_request",
              "requester_number_of_posts_on_raop_at_retrieval",
              "requester_number_of_subreddits_at_request",
              "requester_upvotes_minus_downvotes_at_request",
              "requester_upvotes_minus_downvotes_at_retrieval",
              "requester_upvotes_plus_downvotes_at_request",
              "requester_upvotes_plus_downvotes_at_retrieval",
              "requester_number_of_posts_on_raop_at_request")

for(i in cont_var){
    train[,i] <- as.numeric(train[,i])
}

log_var <- c("requester_received_pizza")

require(car)

for(i in log_var){
  train[,i] <- recode(train[,i],"'FALSE'=0;'TRUE'=1") 
}

# It's yours to take from here!
library(rms)
require(randomForest)

vv <- randomForest(requester_received_pizza ~ 
                     requester_number_of_subreddits_at_request + 
                     requester_number_of_posts_on_raop_at_request + requester_number_of_posts_at_request,
                   data=train)

print(vv)
round(importance(vv), 2)
plot(vv,log="y")

require(rpart)

fit <- rpart(requester_received_pizza ~ 
               requester_account_age_in_days_at_request +
               requester_days_since_first_post_on_raop_at_request +
               requester_number_of_comments_at_request +
               requester_number_of_comments_in_raop_at_request +
               requester_number_of_posts_at_request +
               requester_number_of_subreddits_at_request +
               requester_upvotes_minus_downvotes_at_request +
               requester_upvotes_plus_downvotes_at_request,
               method="anova", data=train)

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# create additional plots 
par(mfrow=c(1,2)) # two plots on one page 
rsq.rpart(fit) # visualize cross-validation results  	

# plot tree 
plot(fit, uniform=TRUE, 
     main="Regression Tree for Mileage ")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

library(tm)
library(topicmodels)
library(XML)

data("JSS_papers", package = "corpus.JSS.papers")
JSS_papers <- JSS_papers[JSS_papers[,"date"] < "2010-08-05",]
JSS_papers <- JSS_papers[sapply(JSS_papers[, "description"],
                                   Encoding) == "unknown",]


remove_HTML_markup <-
  function(s) tryCatch({
    doc <- htmlTreeParse(paste("<!DOCTYPE html>", s),
                           asText = TRUE, trim = FALSE)
    xmlValue(xmlRoot(doc))
    }, error = function(s) s)

corpus <- Corpus(VectorSource(sapply(JSS_papers[, "description"],
                                        remove_HTML_markup)))

Sys.setlocale("LC_COLLATE", "C")

JSS_dtm <- DocumentTermMatrix(corpus,
                                control = list(stemming = TRUE, stopwords = TRUE, minWordLength = 3,
                                                 removeNumbers = TRUE, removePunctuation = TRUE))
dim(JSS_dtm)

dim(JSS_dtm)

library("slam")

summary(col_sums(JSS_dtm))

term_tfidf <-
  tapply(JSS_dtm$v/row_sums(JSS_dtm)[JSS_dtm$i], JSS_dtm$j, mean) *
  log2(nDocs(JSS_dtm)/col_sums(JSS_dtm > 0))
summary(term_tfidf)

JSS_dtm <- JSS_dtm[,term_tfidf >= 0.1]
JSS_dtm <- JSS_dtm[row_sums(JSS_dtm) > 0,]
summary(col_sums(JSS_dtm))

dim(JSS_dtm)

library("topicmodels")

k <- 8
SEED <- 2010
jss_TM <- LDA(JSS_dtm, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000, thin = 100, iter = 1000))
              
terms(jss_TM,15)

topics_v24 <-
      topics(jss_TM)[grep("v024", lapply(JSS_papers[, "identifier"],function(x)x[1]))]

most_frequent_v24 <- which.max(tabulate(topics_v24))

terms(jss_TM, 10)[, most_frequent_v24]
              





####OWN

data("JSS_papers", package = "corpus.JSS.papers")
JSS_papers <- as.character(train$request_text_edit_aware)
# JSS_papers <- JSS_papers[sapply(JSS_papers[, "description"],
                                # Encoding) == "unknown",]


remove_HTML_markup <-
  function(s) tryCatch({
    doc <- htmlTreeParse(paste("<!DOCTYPE html>", s),
                         asText = TRUE, trim = FALSE)
    xmlValue(xmlRoot(doc))
  }, error = function(s) s)

corpus <- Corpus(VectorSource(sapply(JSS_papers,remove_HTML_markup)))

Sys.setlocale("LC_COLLATE", "C")

JSS_dtm <- DocumentTermMatrix(corpus,
                              control = list(stemming = TRUE, stopwords = TRUE, minWordLength = 3,
                                             removeNumbers = TRUE, removePunctuation = TRUE))
dim(JSS_dtm)

dim(JSS_dtm)

library("slam")

summary(col_sums(JSS_dtm))

term_tfidf <-
  tapply(JSS_dtm$v/row_sums(JSS_dtm)[JSS_dtm$i], JSS_dtm$j, mean) *
  log2(nDocs(JSS_dtm)/col_sums(JSS_dtm > 0))
summary(term_tfidf)

JSS_dtm <- JSS_dtm[,term_tfidf >= 0.1]
JSS_dtm <- JSS_dtm[row_sums(JSS_dtm) > 0,]
summary(col_sums(JSS_dtm))

dim(JSS_dtm)

library("topicmodels")

k <- 8
SEED <- 2010
jss_TM <- LDA(JSS_dtm, k = k, method = "Gibbs", control = list(seed = SEED, burnin = 1000, thin = 100, iter = 1000))

terms(jss_TM,20)

hij <- data.frame(topics(jss_TM))
hij$ID <- as.numeric(rownames(hij))
names(hij) <- c("top","ID")

most_frequent_v24 <- which.max(tabulate(topics_v24))

terms(jss_TM, 10)[, most_frequent_v24]



zij <- train

zij$ID <- as.numeric(rownames(zij))

wij <- merge(zij,hij,all.x = TRUE,by="ID")

prop_top <- posterior(jss_TM)$topics

for(i in 1:k){
  idhot <- names(rev(sort(prop_top[,i]))[1:3])
  texhot <- wij[sapply(wij$ID, function(x) any(x == as.numeric(idhot))),"request_text_edit_aware"]
  texhot <- as.character(texhot)
  cat("Topic ",i,": V1\n", sep="")
  cat(texhot[1],"\n")
  cat("Topic ",i,": V2\n", sep="")
  cat(texhot[2],"\n")
  cat("Topic ",i,": V3\n", sep="")
  cat(texhot[3],"\n \n")
}

library("ldatuning")

result <- FindTopicsNumber(
  JSS_dtm,
  topics = seq(from = 10, to = 30, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(result)


dir()

nsfw <- readChar("nsfw_r.txt", file.info("nsfw_r.txt")$size)

require(stringr)

nsfw <- str_split(nsfw," ")[[1]]
nsfw <- nsfw[nchar(nsfw) > 0]

train$requester_subreddits_at_request

train$nsfw_sub <- 0

for(i in 1:nrow(train)){
  all_sub <- train[i,"requester_subreddits_at_request"]
  if(!is.na(all_sub)){
    all_sub <- str_split(all_sub,"; ")[[1]]
    if(any(sapply(all_sub, function(x) any(x==nsfw)))) train[i,"nsfw_sub"] <- 1
  }
}

ddply(train, .(nsfw_sub), summarize,  PizzaR=mean(requester_received_pizza))


test.topics <- posterior(jss_TM,)

train$date <- as.POSIXct(as.numeric(train$unix_timestamp_of_request_utc), origin="1970-01-01",tz = "US/Central")

train$time <- hour(train$date) + minute(train$date) / 60
train$day  <- wday(train$date,label=FALSE)

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

train$season <- getSeason(train$date)

train$title_nchar <- nchar(train$request_title)
train$textr_nchar <- nchar(train$request_text_edit_aware)

train$imgemb_ev <- as.numeric(str_detect(train$request_text_edit_aware,"imgur") | str_detect(train$request_text_edit_aware,"\\["))

gg_time <- ggplot(train, aes(x=time,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ time, lowess = TRUE, data=train)

gg_day <- ggplot(train, aes(x=day,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ day, lowess = TRUE, data=train)

gg_daytime <- ggplot(train, aes(x=time,y=requester_received_pizza,color=day)) +
  histSpikeg(requester_received_pizza ~ time + day, lowess = TRUE, data=train)

gg_seastime <- ggplot(train, aes(x=time,y=requester_received_pizza,color=season)) +
  histSpikeg(requester_received_pizza ~ time + season, lowess = TRUE, data=train)

gg_titlen <- ggplot(train, aes(x=title_nchar,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ title_nchar, lowess = TRUE, data=train)

gg_textrn <- ggplot(train, aes(x=textr_nchar,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ textr_nchar, lowess = TRUE, data=train)

gg_reddm <- ggplot(train, aes(x=requester_number_of_subreddits_at_request,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ requester_number_of_subreddits_at_request, lowess = TRUE, data=train)

gg_comments <- ggplot(train, aes(x=requester_number_of_comments_at_request,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ requester_number_of_comments_at_request, lowess = TRUE, data=train)

gg_commentsraop <- ggplot(train, aes(x=requester_number_of_comments_in_raop_at_request,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ requester_number_of_comments_in_raop_at_request, lowess = TRUE, data=train)

gg_requester_account_age_in_days_at_request <- ggplot(train, aes(x=requester_account_age_in_days_at_request,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ requester_account_age_in_days_at_request, lowess = TRUE, data=train)

# gg_requester_upvotes_minus_downvotes_at_request <- ggplot(train, aes(x=requester_upvotes_minus_downvotes_at_request,y=requester_received_pizza)) +
#   histSpikeg(requester_received_pizza ~ requester_upvotes_minus_downvotes_at_request, lowess = TRUE, data=train)

gg_requester_days_since_first_post_on_raop_at_request <- ggplot(train, aes(x=requester_days_since_first_post_on_raop_at_request,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ requester_days_since_first_post_on_raop_at_request, lowess = TRUE, data=train)

gg_requester_number_of_posts_on_raop_at_request <- ggplot(train, aes(x=requester_number_of_posts_on_raop_at_request,y=requester_received_pizza)) +
  histSpikeg(requester_received_pizza ~ requester_number_of_posts_on_raop_at_request, lowess = TRUE, data=train)

  
  

s <- summary ( requester_received_pizza ~ img_ev + emb_ev + nsfw_sub + season + requester_number_of_posts_on_raop_at_request, data = train )
plot (s , main = "henk", subtitles = FALSE ) 
