#Download preprocessed fine-tuning dataset from drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oV3MHiCRCX-570f_LQoBz56zSA53Mo1M' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oV3MHiCRCX-570f_LQoBz56zSA53Mo1M" -O Data_Challenge_new.zip && rm -rf /tmp/cookies.txt

# Download dataset from Dropbox
#wget https://www.dropbox.com/s/8yf786e7en42qnt/Data_Place.zip?dl=1

#mkdir Data_Places

# Unzip the downloaded zip file
unzip ./Data_Challenge_new.zip

# Remove the downloaded zip file
rm ./Data_Challenge_new.zip

# Remove the MACOSX folder
rm -rf ./__MACOSX/
