import matplotlib.pyplot as plt
# IMPORTANT must run files in order: LinearModel.py, NNModel.py, XGModel.py, Accuracies.py

# Reads the text file 'r2_file.txt' which contains the R2_scores for all models
# Cleans up data, prepare for plotting by converting string -> float again
with open("r2_file.txt", "r") as file:
    fr1 = file.read()  
    test_data = fr1.split(",")
test_data = [item for item in test_data if item]
r2_test = [float(item) for item in test_data]

with open("r2_val_file.txt", "r") as file:
    fr2 = file.read()  
    val_data = fr2.split(",")
val_data = [item for item in val_data if item]
r2_val = [float(item) for item in val_data]

with open("r2_train_file.txt", "r") as file:
    fr3 = file.read()  
    train_data = fr3.split(",")
train_data = [item for item in train_data if item]
r2_train = [float(item) for item in train_data]


x_axis = [1, 2, 3]
x_labels = ["Model 1", "Model 2", "Model 3"] 
plt.figure(figsize=(12, 6))
# R2 Scores for test
plt.scatter(x_axis, r2_test, color="orange", linewidth=2)
plt.plot(x_axis, r2_test, label="R2 Test", color="orange", linewidth=2)
# R2 Scores for validation
plt.scatter(x_axis, r2_val, color="red", linewidth=2)
plt.plot(x_axis, r2_val, label="R2 Validation", color="red", linewidth=2)
# R2 Scores for training
plt.scatter(x_axis, r2_train, color="Blue", linewidth=2)
plt.plot(x_axis, r2_train, label="R2 Training", color="Blue", linewidth=2)

plt.xticks(ticks=x_axis, labels=x_labels)
plt.title("Training, Test, and Validation R2 Score for Different Models")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.legend()
plt.grid(True)
plt.show()
