import numpy as np
from PIL import Image
import fitz
from pytesseract import image_to_data, Output
from io import BytesIO
import pandas as pd


def apply_smth(df: pd.DataFrame):
    to_be_merged = []
    target_words = 4
    remaining_words = df.shape[0] - target_words
    # df = df.filter(lambda x: x["par_num"] < 3)
    df = df.reset_index()
    if remaining_words == 0:
        return df["text"]
    if remaining_words < 0:
        return
    for i in range(remaining_words):
        to_be_merged.append(df.loc[i, "text"].capitalize())
        if i != 0:
            df = df.drop(i)
    df.loc[0, "text"] = "".join(to_be_merged)
    return df["text"]


def crop_pdf(passed_page, start_word: str, end_word: str, dpi=300):
    pg = passed_page
    page_width = round(pg.rect.width)
    start_rect = pg.get_textpage_ocr(3, "pol", dpi, True).search(start_word)[0].rect
    end_rect = pg.get_textpage_ocr(3, "pol", dpi, True).search(end_word)[0].rect
    start_y = start_rect.round().bottom_left[1]
    end_y = end_rect.round().top_left[1]
    pg.set_cropbox(fitz.Rect(0, start_y, page_width, end_y))
    return pg


def create_dataframe(file_or_path):
    """
    This function takes a pdf file and converts it to an image and then extracts text using OCR and saves it into a dataframe.

    :param file_or_path: str or opened file
        The full path of the pdf file in the 'input' directory, or an already opened file.
    :return: DataFrame
        A dataframe containing OCR extracted text from the image.
    """
    if isinstance(file_or_path, str):
        # If a string is provided, it is treated as a path to a pdf file
        doc = fitz.open(file_or_path)
    elif hasattr(file_or_path, "read"):
        # If an object with a 'read' attribute is provided, it is treated as an opened file
        doc = fitz.open("pdf", file_or_path.read())
    else:
        raise ValueError(
            "Invalid input. Expected either a path to a pdf or an opened file"
        )
    page = list(doc)[0]  # Get first page of the pdf
    pix = page.get_pixmap(
        dpi=300, colorspace=fitz.csGRAY
    )  # Convert page to grayscale image
    pix.invert_irect(pix.irect)  # Invert color of the image
    pdf_bytes = pix.tobytes()  # Convert image to bytes
    img = Image.open(BytesIO(pdf_bytes))  # Open image from bytes

    # Use pytesseract to extract text and save it into a dataframe
    df = image_to_data(img, "pol", config="--psm 4", output_type=Output.DATAFRAME)

    doc.close()

    return df


df = create_dataframe("input/a-2.pdf")
df_copy = df.copy()
df_copy = df_copy.drop(["width", "height", "level", "left", "top", "conf"], axis=1)
df_copy = df_copy.dropna(subset="text")
df_copy["text"] = df_copy["text"].str.lower()
shop = "auchan" if "auchan" in df_copy["text"].array else "biedronka"
main_index = df_copy[df_copy["text"] == "niefiskalny"]["block_num"].values[0]
df_copy = df_copy[df_copy["block_num"] == main_index]
df_copy_copy = df_copy.copy()
df_copy_copy = df_copy_copy.drop(["block_num"], axis=1)
df_copy_copy = df_copy_copy[df_copy_copy["par_num"] == 1]
gbl = df_copy_copy.groupby("line_num")
gbl = gbl.filter(lambda x: x["line_num"].count() > 2)
gbl = gbl.groupby("line_num")
gbl = gbl.apply(apply_smth, include_groups=False)
gbl = gbl.reset_index()
gbl: pd.Series = gbl[0].dropna()
np_array = gbl.values
indices = np.arange(5, len(np_array), 5)
sub_arrays = np.split(np_array, indices)
final_dframe = pd.DataFrame(
    sub_arrays, columns=["name", "amount", "star", "price_per_unit", "price"]
)
final_dframe = final_dframe.dropna()
final_dframe = final_dframe.map(lambda x: x.replace(",", "."))
index = 0
try:
    index = final_dframe[final_dframe["price_per_unit"] < "0"].index[0]
except IndexError:
    index = len(final_dframe["price"])
    pass

final_dframe = final_dframe[:index]
final_dframe = final_dframe.drop(["star"], axis=1)
try:
    tax_index = final_dframe[
        final_dframe["name"].str.lower().str.replace(".", "") == "sprzed"
    ].index[0]
except IndexError:
    tax_index = len(final_dframe["name"])
final_dframe["amount"] = final_dframe["amount"].str.replace("l", "1")
final_dframe = final_dframe[:tax_index]
