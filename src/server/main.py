import json
from typing import Annotated, List, Literal
from fastapi import FastAPI, UploadFile, File, Form
from transformers import Receit
from pydantic import BaseModel, PositiveFloat

app = FastAPI()


class Product(BaseModel):
    name: str
    amount: PositiveFloat
    price_per_unit: PositiveFloat
    price: PositiveFloat



class Response(BaseModel):
    shop: Literal["auchan", "biedronka"]
    data: List[Product]


@app.post("/uploadfile/")
async def create_upload_file(
    file: Annotated[UploadFile, File()],
    shop_name: Annotated[Literal["auchan", "biedronka"], Form()],
) -> Response:

    data = await Receit.create_json(file, shop_name)

    return {"shop": shop_name, "data": json.loads(data)}
