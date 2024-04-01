import pandas as pd
from PIL import Image
from io import BytesIO, StringIO
from typing import List, Tuple, Optional
from pytesseract import image_to_data, Output
import fitz
import numpy as np


class DataFrameCreator:
    """
    A class that creates a pandas dataframe from a PDF file or a CSV file.

    :param path_or_file: str or opened file
        The full path of the pdf/csv file in the 'input' directory, or an already opened file.
    """

    def __init__(self, path_or_file):
        self._df = None
        self._doc = None
        self._page = None
        self._page2 = None
        self._img = None
        if isinstance(path_or_file, str):
            # If a string is provided, it is treated as a path to a pdf file
            doc = fitz.open(path_or_file)
        elif hasattr(path_or_file, "read"):
            # If an object with a 'read' attribute is provided, it is treated as an opened file
            doc = fitz.open("pdf", path_or_file.read())
        else:
            raise ValueError(
                f"Invalid input. Expected either a path to a pdf or an opened file, got: {type(path_or_file)}"
            )
        self._doc = doc

    class PrepareReceiptData:
        @staticmethod
        def add_nulls(search: list):
            discount_idxs = []
            array = search.copy()
            for i, word in enumerate(array):
                if word == "Rabat":
                    discount_idxs.append(i)
            for index in reversed(discount_idxs):
                array[index + 1 : index + 1] = ["0", "0"]
            return array

        @staticmethod
        def crop_pdf(passed_page: fitz.Page, start_word: str, end_word: str):
            page_width = round(passed_page.rect.width)
            start_rect = (
                passed_page.get_textpage_ocr(3, "pol", 300, True)
                .search(start_word)[0]
                .rect
            )
            end_rect = (
                passed_page.get_textpage_ocr(3, "pol", 300, True)
                .search(end_word)[0]
                .rect
            )
            start_y = start_rect.round().bottom_left[1]
            end_y = end_rect.round().top_left[1]
            passed_page.set_cropbox(fitz.Rect(0, start_y, page_width, end_y))

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        if isinstance(value, pd.DataFrame):
            self._df = value
        else:
            raise ValueError("Value must be a pandas DataFrame")

    def __create_auchan_dataframe(self) -> pd.DataFrame:
        """
        This function takes a pdf file and converts it to an image and then extracts text using OCR and saves it into a dataframe.

        :param file_or_path: str or opened file
            The full path of the pdf file in the 'input' directory, or an already opened file.
        :return: DataFrame
            A dataframe containing OCR extracted text from the image.
        """

        page = self._doc[0]  # Get first page of the pdf

        self._page = page
        self._page2 = "xd"

        pix = page.get_pixmap(
            dpi=300, colorspace=fitz.csGRAY
        )  # Convert page to grayscale image
        pix.invert_irect(pix.irect)  # Invert color of the image
        pdf_bytes = pix.tobytes()  # Convert image to bytes
        img = Image.open(BytesIO(pdf_bytes))  # Open image from bytes

        # Use pytesseract to extract text and save it into a dataframe
        df = image_to_data(img, "pol", config="--psm 4", output_type=Output.DATAFRAME)

        self._doc.close()

        self._df = df

    def __create_biedronka_dataframe(self) -> pd.DataFrame:
        if len(self._doc) == 0:
            raise ValueError("Provided pdf has 0 pages")
        if len(self._doc) > 1:
            raise ValueError(
                f"Receipt is too big, make sure that biedronka has only 1 page receipt, given receipt has {len(self._doc)} pages"
            )
        page = self._doc[0]
        self.PrepareReceiptData.crop_pdf(page, "NIEFISKALNY", "Sprzed")
        text_page = page.get_textpage_ocr(3, "pol", 300, True)
        text = text_page.extractText()
        lines = text.split("\n")
        lines = self.PrepareReceiptData.add_nulls(lines)
        grouped_lines = [";".join(lines[i : i + 5]) for i in range(0, len(lines), 5)]
        grouped_lines = self.PrepareReceiptData.add_nulls(grouped_lines)
        result = "\n".join(grouped_lines)
        result = result.encode("utf-8").decode()
        result = StringIO(result)
        df = pd.read_table(result, sep=";")

        self._doc.close()
        self._df = df

    def create_dataframe(self, shop_name):
        if shop_name == "auchan":
            self.__create_auchan_dataframe()
        elif shop_name == "biedronka":
            self.__create_biedronka_dataframe()
        else:
            raise ValueError("No shop name was provided")


class ReceiptRefactor:
    def __init__(self, df):
        self._df = df

    class DataTransform:
        @staticmethod
        def clean_receipt_data(df: pd.DataFrame) -> pd.DataFrame:
            """
            This function cleans the dataframe by dropping unnecessary columns and converting all texts to lower case.
            Args:
                receipt_df (pd.DataFrame): The dataframe to be cleaned.

            Returns:
                pd.DataFrame: The cleaned dataframe.
            """
            new_df = df.copy()
            new_df.drop(
                ["width", "height", "level", "left", "top", "conf"],
                axis=1,
                inplace=True,
            )
            new_df.dropna(subset="text", inplace=True)
            new_df["text"] = new_df["text"].str.lower()
            return new_df

        @staticmethod
        def adjust_price(df: pd.DataFrame):
            new_df = df.copy()
            new_df["amount"] = (
                new_df["amount"].replace("l", "1").astype(float)
            )  # Ensure 'amount' is float
            new_df["price_per_unit"] = new_df["price_per_unit"].astype(
                float
            )  # Ensure 'price_per_unit' is float

            # Multiply 'amount' by 'price_per_unit' and assign it to 'price' column
            new_df["price"] = new_df["amount"].mul(new_df["price_per_unit"])
            return new_df

        @staticmethod
        def sum_discount(arg_pd: pd.DataFrame):
            dfp = arg_pd.copy()
            x = dfp[dfp["Nazwa"] == "Rabat"]
            for i in x.index.array:
                main_price = dfp.loc[i - 1, "Wartość"]
                discount_price = x["Cena"][i]
                new_price = np.round(main_price + discount_price, decimals=2)
                dfp.loc[i - 1, "Wartość"] = new_price
            dfp = dfp.drop(x.index.array)
            return dfp

    class UtilGetters:
        @staticmethod
        def get_shop_name(receipt_df: pd.DataFrame) -> str:
            """
            This function determines the shop name from the dataframe.

            Args:
                receipt_df (pd.DataFrame): The dataframe containing receipt data.

            Returns:
                str: The name of the shop.
            """
            return "auchan" if "auchan" in receipt_df["text"].array else "biedronka"

        @staticmethod
        def get_main_block_number(df: pd.DataFrame) -> int:
            """
            This function gets the main block number from the dataframe.

            Args:
                receipt_df (pd.DataFrame): The dataframe containing receipt data.

            Returns:
                int: The main block number.
            """
            return df[df["text"] == "niefiskalny"]["block_num"].values[0]

        @staticmethod
        def get_grouped_by_line(df: pd.DataFrame) -> pd.DataFrame:
            """
            This function groups the dataframe by line.

            Args:
                cleaned_df (pd.DataFrame): The cleaned dataframe.

            Returns:
                pd.DataFrame: The grouped dataframe.
            """
            new_df = df.copy()
            new_df = new_df[new_df["par_num"] == 1]
            new_df = new_df.groupby("line_num").filter(
                lambda x: x["line_num"].count() > 2
            )
            return new_df

        @staticmethod
        def __merge_and_capitalize_text(df: pd.DataFrame) -> Optional[pd.Series]:
            """
            This function takes a dataframe with a 'text' column and merges the first n-4 rows into one,
            capitalizing each word in the merged text. If there are less than 4 words in the text, it returns None.

            :param df: DataFrame containing 'text' column to be manipulated.
            :return: Modified Series of text or None if there are less than 4 words.
            """
            target_words = 4
            remaining_words = df.shape[0] - target_words

            # Check for the case where there are less than 4 words, return None in such a case
            if remaining_words < 0:
                return None

            df = df.reset_index()

            # If there are exactly 4 or more words in text, merge and capitalize
            to_be_merged = []
            for i in range(remaining_words):
                to_be_merged.append(df.loc[i, "text"].capitalize())
                if i != 0:
                    df = df.drop(i)

            # Join the words and update the first row of 'text' column
            df.loc[0, "text"] = "".join(to_be_merged)

            return df["text"]

        @classmethod
        def get_sub_arrays(cls, df: pd.DataFrame) -> List[np.array]:
            """
            This function gets sub arrays from the grouped dataframe.

            Args:
                grouped_by_line (pd.DataFrame): The grouped dataframe.

            Returns:
                List[np.array]: The list of sub arrays.
            """
            new_df = df.copy()
            new_df = (
                new_df.groupby("line_num")
                .apply(cls.__merge_and_capitalize_text, include_groups=False)
                .reset_index()
            )
            new_df = new_df[0].dropna()
            return np.split(new_df.values, np.arange(5, len(new_df.values), 5))

        @staticmethod
        def get_invalid_price_index(df: pd.DataFrame) -> int:
            """
            This function gets the index of invalid price from the dataframe.

            Args:
                self.df (pd.DataFrame): The final dataframe.

            Returns:
                int: The index of invalid price.
            """
            try:
                return df[df["price_per_unit"] < "0"].index[0]
            except IndexError:
                return len(df["price"])

        def get_tax_index(df: pd.DataFrame) -> int:
            """
            This function gets the index of tax from the dataframe.

            Args:
                self.df (pd.DataFrame): The final dataframe.

            Returns:
                int: The index of tax.
            """
            try:
                return df[
                    df["name"].str.lower().str.replace(".", "") == "sprzed"
                ].index[0]
            except IndexError:
                return len(df["name"])

    @property
    def df(self):
        return self._df

    def __refactor_auchan_receipt_data(self) -> Tuple[str, int, int, pd.DataFrame]:
        """
        This function refactors the receipt data and returns the shop name, main block number, tax index and the final dataframe.

        Args:
            receipt_df (pd.DataFrame): The dataframe containing receipt data.

        Returns:
            Tuple[str, int, int, pd.DataFrame]: The shop name, main block number, tax index and the final dataframe.
        """
        df = self._df
        df = self.DataTransform.clean_receipt_data(df)
        main_block_number = self.UtilGetters.get_main_block_number(df)
        df = df[df["block_num"] == main_block_number].drop(["block_num"], axis=1)
        df = self.UtilGetters.get_grouped_by_line(df)
        sub_arrays = self.UtilGetters.get_sub_arrays(df)
        df = pd.DataFrame(
            sub_arrays, columns=["name", "amount", "star", "price_per_unit", "price"]
        ).dropna()
        df = df.map(lambda x: x.replace(",", "."))
        invalid_price_index = self.UtilGetters.get_invalid_price_index(df)
        df = df[:invalid_price_index].drop(["star"], axis=1)
        tax_index = self.UtilGetters.get_tax_index(df)
        df = df[:tax_index]
        df = self.DataTransform.adjust_price(df)
        self._df = df

    def __refactor_biedronka_receipt_data(self):
        df = self._df
        df["Ilość"] = df["Ilość"].map(lambda row: row.split(" x")[0])
        ints = df[["Ilość", "Cena", "Wartość"]]
        ints = ints.map(lambda series: series.replace(",", ".")).map(np.float32)
        df.update(ints)
        df = self.DataTransform.sum_discount(df)
        df = df.reset_index(drop=True)
        df.drop("PTU", inplace=True, axis=1)
        df.rename(
            columns={
                "Nazwa": "name",
                "Ilość": "amount",
                "Cena": "price_per_unit",
                "Wartość": "price",
            },
            inplace=True,
        )
        self._df = df

    def refactor_receipt_data(self, shop_name):
        if shop_name == "auchan":
            self.__refactor_auchan_receipt_data()
        elif shop_name == "biedronka":
            self.__refactor_biedronka_receipt_data()
        else:
            raise ValueError("Provided shop name is not correct")


async def create_json(file, shop_name):
    b = BytesIO(await file.read())

    creator = DataFrameCreator(b)
    creator.create_dataframe(shop_name)

    refactor = ReceiptRefactor(creator.df)
    refactor.refactor_receipt_data(shop_name)

    data = refactor.df.to_json(None, orient="records")

    return data


if __name__ == "__main__":
    try:
        with open("input/biedronka/2.pdf", "rb") as f:
            b = BytesIO(f.read())
            creator = DataFrameCreator(b)
            creator.create_dataframe("biedronka")

            refactor = ReceiptRefactor(creator.df)
            refactor.refactor_receipt_data("biedronka")

            dictionary = refactor.df.to_dict("records")

            print(dictionary)
            # refactor.df.to_json("output/biedronka/2.json")
    except FileNotFoundError:
        print("The file could not be found.")
