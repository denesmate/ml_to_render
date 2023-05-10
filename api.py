from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel
from train_model_personal import train_data, process_data, import_data


# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    random_number: int

# Save statenumbers from POST method in the memory
statenumbers = {}

# Initialize FastAPI instance
app = FastAPI()


# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/statenumbers/")
async def create_item(item: TaggedItem):
    statenumbers[item.random_number] = item
    return item


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Please post your random state number to get inference result!"}

# A GET that in this case just returns the random_number we pass,
# but a future iteration may link the random_number here to the one we defined in our TaggedItem.
@app.get("/statenumbers/{random_number}")
async def get_statenumbers(random_number: int, count: int = 1):
    PTH = "./census.csv"
    data = import_data(PTH)
    X_train, X_val, X_test, y_train, y_val, X_train_describe = process_data(data, random_state_number=random_number)

    return train_data(X_train, X_val, X_test, y_train, y_val, X_train_describe)

