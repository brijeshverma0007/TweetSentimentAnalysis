------------------------------------------------------------------------------------------------------------------
Running on local machine with Jupyter notebook
------------------------------------------------------------------------------------------------------------------
Put the jupyter notebook file and the 5 csv files in a folder.

The csv files are Company.csv, Company_Tweet.csv, CompanyValue.csv, Tweet.csv, sample_train_raw_tweet_db.csv

Link for csv files: https://drive.google.com/drive/folders/1K1ZHSeN37nDNc8HL8vU9xAqQfL-LEnCC?usp=sharing

Open the jupyter notebook file.

Make sure python and pip is installed in the laptop. pip is needed to download packages which are required.

All the packages that need to be installed are put in the codes as pip install which will download the packages.

Before running the code make sure to change the path to the directory where csv files are places.

Code for doing this: %cd /Desktop/SMDMProject/

Run the code snippets one by one or can run the whole notebook at once.

sample_train_raw_tweet_db.csv file is the sample file taken and the sentiment has been manually given for the tweets in the sample data.
This is used for checking the accuracy of vader and textblob.
As we have done it manually, we have given the csv file which needs to be put in the directory to load it to a data frame in the code and check accuracy.


------------------------------------------------------------------------------------------------------------------
Running with Google Colab
------------------------------------------------------------------------------------------------------------------
Upload the Jupyter notebook file to the drive and also the 5 csv files.

The csv files are big in size and might take time to upload. An alternative and easier menthod is to add shortcut to drive from the link below instead of downloading and uploading them again.

The csv files are Company.csv, Company_Tweet.csv, CompanyValue.csv, Tweet.csv, sample_train_raw_tweet_db.csv

Link for the Google Colab: https://colab.research.google.com/drive/1QMJDU51aGP1tOZkCzKgnjA9hkr-VHIim?usp=sharing 

Link for csv files: https://drive.google.com/drive/folders/1K1ZHSeN37nDNc8HL8vU9xAqQfL-LEnCC?usp=sharing

Open the jupyter notebook file in Google Colab.

All the packages that need to be installed are put in the codes as pip install which will download the packages.

Before running the code make sure to change the path to the directory where csv files are places.

Mount the drive to the coloboratory notebook with the command in the notebook.

Code for doing this: %cd /drive/MyDrive/SMDMProject/

Run the code snippets one by one or can run the whole notebook at once.

sample_train_raw_tweet_db.csv file is the sample file taken and the sentiment has been manually given for the tweets in the sample data.
This is used for checking the accuracy of vader and textblob.
As we have done it manually, we have given the csv file which needs to be put in the directory to load it to a data frame in the code and check accuracy.