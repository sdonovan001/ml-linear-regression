import pandas as pd
import wrapper.model as wpr

def prompt_user(learning_rate, epochs, batch_size):
   learning_rate = float(input("Learning Rate [{}]:".format(learning_rate)) or learning_rate)
   epochs = int(input("Epochs [{}]:".format(epochs)) or epochs)
   batch_size = int(input("Batch Size [{}]:".format(batch_size)) or batch_size)

   return learning_rate, epochs, batch_size

########### main ##############
# hyperparameters used for model training
learning_rate = 0.5
epochs = 20
batch_size = 50

features = ['TRIP_MILES']
label = 'FARE'

training_data = pd.read_csv("./datasets/training-data.csv")

model = None
save_model = "N" 

while save_model == "N":
   model = wpr.Model(training_data, features, label)

   learning_rate, epochs, batch_size = prompt_user(learning_rate, epochs, batch_size)

   model.set_params(learning_rate, epochs, batch_size)
   model.train()
   model.graph()

   save_model = input("\nSave model and exit? [N]:") or "N" 

model.save("./saved_model/fare-model/1")
