import pickle

aaa = {"name": "jack", 123: "No"}
save_file = open("1.txt", "wb")
pickle._dump(aaa, save_file)
save_file.close()

load_file = open("1.txt", "rb")
bbb = pickle.load(load_file)
load_file.close()
print(bbb)
