from io import BytesIO
import json
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Form
from transformers import Receit

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(
    file: Annotated[UploadFile, File()], shop_name: Annotated[str, Form()]
):
    creator = Receit.DataFrameCreator(BytesIO(await file.read()))
    refactor = Receit.ReceiptRefactor(creator.df)
    refactor.refactor_receipt_data(shop_name)

    data = refactor.df.to_json()

    return {"shop": shop_name, "data": json.dumps(data)}
