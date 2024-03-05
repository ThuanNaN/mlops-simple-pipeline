import os
import mlflow
from dotenv import load_dotenv
import torch
import torch.nn as nn   
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
load_dotenv()
from utils import Log

logger = Log(__file__).get_logger()
logger.info("Trainer")


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
except Exception as e:
    logger.error(f"Error: {e}")
    raise e

class Trainer:
    def __init__(self, 
                 model, 
                 num_epochs, 
                 learning_rate, 
                 weight_decay, 
                 train_data, 
                 val_data, 
                 batch_size: int,
                 best_model_metric: str,
                 device: str, 
                 mlflow_log_pamrams, 
                 verbose=False
                 ) -> None:
        
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size    
        self.best_model_metric = best_model_metric
        self.device = device
        self.mlflow_log_pamrams = mlflow_log_pamrams
        self.verbose = verbose  
    
    def train(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        run_name = f"{self.mlflow_log_pamrams['model_name']}_{self.mlflow_log_pamrams['data_version']}"

        with mlflow.start_run(run_name=run_name) as run:

            mlflow.log_params({
                "optimizer": optimizer.__class__.__name__,
                "criterion": self.criterion.__class__.__name__,
            })
            mlflow.log_params(self.mlflow_log_pamrams)

            # with open("model_summary.txt", "w") as f:
            #     f.write(str(summary(self.model)))
            # mlflow.log_artifact("model_summary.txt")

            best_val_loss = float('inf')
            best_val_acc = float('-inf')
            best_val_loss_state_dict = None
            best_val_acc_state_dict = None

            
            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss = 0.0
                running_corrects = 0
                running_total = 0
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    running_corrects += torch.sum(predicted == labels.data)
                    running_total += labels.size(0)
                
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = running_corrects / running_total

                mlflow.log_metric("Training_loss", f"{epoch_loss:2f}", step=epoch)
                mlflow.log_metric("Training_acc", f"{epoch_acc:2f}", step=epoch)
        
                val_loss,  val_acc = self.validate(val_loader, epoch=epoch)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_state_dict = self.model.state_dict()
                    
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_acc_state_dict = self.model.state_dict()

                if self.verbose:
                    logger.info(f"Epoch [{epoch + 1}]/{self.num_epochs} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            mlflow.log_metric("Best_val_loss", best_val_loss)
            mlflow.log_metric("Best_val_acc", best_val_acc)

            if self.best_model_metric == "val_loss":
                best_model_state_dict = best_val_loss_state_dict
            else:
                best_model_state_dict = best_val_acc_state_dict
                
            self.model.load_state_dict(best_model_state_dict)
            mlflow.pytorch.log_model(self.model, "model")


    def validate(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        with torch.no_grad():
            for (inputs, labels) in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                running_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                running_corrects += (predicted == labels).sum().item()
                running_total += labels.size(0)
        val_loss = running_loss / len(val_loader)
        val_acc = running_corrects / running_total  
        mlflow.log_metric("Val_loss", f"{val_loss:2f}", step=epoch)
        mlflow.log_metric("Val_acc", f"{val_acc:2f}", step=epoch)
        return val_loss, val_acc
    

    def test(self, test_data):
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return self.validate(test_loader, epoch=0)


    def predict(self, image, transform, class_names):
        self.model.eval()
        image = transform(image).unsqueeze(0)        
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
        return predicted_class