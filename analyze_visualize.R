#Read from the database for analysis and display
#Interactive for now but scripted later

rm(list=ls())  # clears environment
try(dev.off())  # clears plots

library(RSQLite)
#library(stringr)
#library(gam)
#library(MASS)
library(ggplot2)
#library(psych)
#library(reshape2)  #for melt
#library(psych)
#library(plyr)
#library(plotly)
library(rbokeh)

wd <- "/Users/epeterson/Documents/RaspberryPi/squash"
setwd(wd)

#load the player database
dbname <- "sqlitedb_complete.sqlite"
con <- dbConnect(drv=RSQLite::SQLite(), dbname=dbname)
# get a list of all tables
alltables_players = dbListTables(con)
players <- dbGetQuery(con, "SELECT * FROM player")
players$Player_ID <- factor(players$Player_ID)
dbDisconnect(con)
names <- data.frame(
  Player_ID=players$Player_ID,
  name=factor(paste(players$First_Name,players$Middle_Name,players$Last_Name)))

#load the ratings database
dbname <- "ratingdb.sqlite"
con <- dbConnect(drv=RSQLite::SQLite(), dbname=dbname)
# get a list of all tables
alltables_ratings <- dbListTables(con)
matches <- dbGetQuery(con, "SELECT * FROM Matches")
EloRatings <- dbGetQuery(con, "SELECT * FROM Elo_Ratings")
EloRatings$Player_ID <- factor(EloRatings$Player_ID)
EloRatings$Date <- as.Date(EloRatings$Date)
dbDisconnect(con)

#get the names
EloNames <- inner_join(EloRatings, names, by="Player_ID")

#plotting (TODO: pick top 20 all time or something)
gplt <- ggplot(data=EloNames[EloNames$Player_ID %in% seq(1,20),], 
               aes(x=Date, y=Rating, group=name, colour=name)) 
gplt <- gplt + geom_line() + geom_point()
gplt <- gplt + scale_x_date()
print(gplt)

fig <- figure(data=EloNames[EloNames$Player_ID %in% seq(1,20),], 
              width=1000, 
              height=600,
              legend_location="bottom_right" )
fig <- ly_lines(fig, x=Date, y=Rating, color=name)
fig <- ly_points(fig, x=Date, y=Rating, color=name, 
                 hover=list(name, Date, Rating))
print(fig)
rbokeh2html(fig, paste(wd, "/ratingplot_R.html", sep=""))
