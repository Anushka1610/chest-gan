import pandas as pd


list_of_classes = ['Atelectasis','No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax']
fields = ['Image Index', 'Finding Labels']

fpath = "~/Downloads/"

csv_path = "Data_Entry_2017.csv"
save_path = "~/Desktop/newDataTest.csv"
u_cols = ['Image Index','Finding Labels']
# load CSV (only the relevant columns)
df = pd.read_csv(fpath + csv_path, usecols=fields)

# kill all rows not containing target classes
df = df[df['Finding Labels'].isin(list_of_classes)]

# initialize an empty count dict
# where to find the 1000th instance of each class
indices = {name: 0 for name in list_of_classes}

"""
Never finds the 1000th image of anything other than 'No Finding'
stores thousandth index incorrectly --> doesn't actually drop any rows
"""

# kill so that only 1k of target classes remain
for id in list_of_classes:
    print("Beginning {}".format(id))

    print("size of dataframe for this ID: {}".format(len(df[df['Finding Labels'] == id].index)))

    count = 1
    for counter, row in df[df['Finding Labels'] == id].iterrows():
        # print(count)
        # This version is Really inefficient but also really simple. Don't wanna deal with weird loc/drop issues rn
        if count >= 801 and count<=1000:
            count += 1
            continue
        else:
            # deleting while iterating -- a bad idea, for sure....
            # https://stackoverflow.com/questions/28876243/how-to-delete-the-current-row-in-pandas-dataframe-during-df-iterrows
            # print("Size of DF before drop: {}".format(len(df.index)))
            df.drop(counter, inplace=True)
            # print("Size of DF after drop: {}".format(len(df.index)))
            count += 1

    print("size of dataframe for this ID (reduced): {}".format(len(df[df['Finding Labels'] == id].index)))

print("Size of DF being saved: {}".format(len(df.index)))
print("Number of No Finding: {}".format(len(df[df['Finding Labels'] == "No Finding"].index)))
# Write out CSV
df.to_csv(save_path)





# In for loop: just do "if count <= 1000 next; else df.drop[row]"
# messy, but might work