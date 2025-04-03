import pandas as pd
import wrapper.model as wpr

def prompt_user(learning_rate, batch_size, epochs):
   learning_rate = float(input("Learning Rate [{}]:".format(learning_rate)) or learning_rate)
   batch_size = int(input("Batch Size [{}]:".format(batch_size)) or batch_size)
   epochs = int(input("Epochs [{}]:".format(epochs)) or epochs)

   return learning_rate, batch_size, epochs

########### main ##############
# hyperparameters used for model training
learning_rate = 0.5
batch_size = 50
epochs = 20

features = ['TRIP_MILES']
label = 'FARE'

training_data = pd.read_csv("./datasets/training-data.csv")

model = None
save_model = "N" 

while save_model == "N":
   model = wpr.Model(training_data, features, label)

   learning_rate, batch_size, epochs = prompt_user(learning_rate, batch_size, epochs)

   model.set_params(learning_rate, batch_size, epochs)
   model.train()
   model.graph()

   save_model = input("\nSave model and exit? [N]:") or "N" 

model.save(model_name="fare-model")
