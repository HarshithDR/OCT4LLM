import wandb
import random
from dotenv import load_dotenv
import os

load_dotenv()
wandb_username = os.getenv("wandb_username")
print("w = ", wandb_username)

class wandb_class:
    link = ""
    
def wandb_initialize_fun(model_name):
    project_name = model_name.replace("/","") + str(random.randint(1,734659))
    run = wandb.init(
        project = project_name,
        entity = wandb_username
    )
    w_class = wandb_class()
    w_class.link = "https://wandb.ai/" + wandb_username + '/' + project_name
    print(w_class.link)
    
    return w_class.link